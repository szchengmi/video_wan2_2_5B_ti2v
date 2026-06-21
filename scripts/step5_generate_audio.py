#!/usr/bin/env python3
"""Step 5: 配音生成 - 多轨合成 (ChatTTS / edge-tts)

支持两种模式：
1. 新模式：读取 audio_events → 多轨合成（dialogue + VO + SFX + Atmos 按 timecode 混音）
2. 旧模式：纯文本 dialogue/narration → 单轨合成（向后兼容）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import subprocess
import wave, struct
import numpy as np
import re


def parse_timecode(tc):
    """解析 '0.5-2.3' → (start_sec, end_sec)"""
    try:
        parts = tc.strip().split("-")
        return float(parts[0]), float(parts[1])
    except:
        return 0.0, 0.0


def load_sound_file(path, target_sr=24000):
    """加载音效/环境音文件，返回 numpy float32 数组"""
    try:
        if not os.path.exists(path):
            return None
        # 用 ffmpeg 解码任意格式
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", os.path.abspath(path),
            "-f", "s16le", "-ac", "1", "-ar", str(target_sr),
            "-v", "error", "pipe:1"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) == 0:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        return samples
    except:
        return None


def generate_voice_segment(text, role, speed_offset, emotion, target_duration, chat, use_chattts):
    """生成单段角色语音，返回 float32 numpy 数组"""
    if use_chattts and chat is not None:
        return _gen_chattts(text, role, speed_offset, target_duration, chat)
    else:
        return _gen_edge_tts(text, role, speed_offset, target_duration)


def _gen_chattts(text, role, speed_offset, target_duration, chat):
    """ChatTTS 生成单段语音"""
    try:
        import torch
        # 角色 seed
        ROLE_SEEDS = {"xiaoming": 42, "xiaoli": 137, "boss_wang": 256, "narrator": 0}
        seed = ROLE_SEEDS.get(role, 0)
        generator = torch.Generator(device=chat.speaker.mean.device)
        generator.manual_seed(seed)
        spk = (
            torch.randn(chat.speaker.dim, device=chat.speaker.std.device,
                        dtype=chat.speaker.std.dtype, generator=generator)
            .mul_(chat.speaker.std)
            .add_(chat.speaker.mean)
        )
        spk_emb = chat.speaker._encode(spk)

        # speed tag: 基础5 + speed_offset
        speed_tag = min(9, max(0, 5 + int(speed_offset)))
        params = ChatTTS.Chat.InferCodeParams(
            prompt=f"[speed_{speed_tag}]",
            temperature=0.3,
            spk_emb=spk_emb,
        )
        wavs = chat.infer([text], params_infer_code=params, skip_refine_text=True)
        if wavs and len(wavs) > 0:
            audio = wavs[0]
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            return audio.astype(np.float32)
    except Exception as e:
        log(f"  [warn] ChatTTS {role}: {e}")
    return None


def _gen_edge_tts(text, role, speed_offset, target_duration):
    """edge-tts fallback"""
    try:
        import asyncio, edge_tts
        EDGE_V = {
            "xiaoming": "zh-CN-YunxiNeural",
            "xiaoli": "zh-CN-XiaoxiaoNeural",
            "boss_wang": "zh-CN-YunjianNeural",
            "narrator": "zh-CN-YunxiNeural"
        }
        vn = EDGE_V.get(role, "zh-CN-YunxiNeural")
        base_speed = {"xiaoming": 0.9, "xiaoli": 1.1, "boss_wang": 0.85, "narrator": 1.0}.get(role, 1.0)
        speed_mult = base_speed * (1.0 + speed_offset * 0.05)
        speed_mult = max(0.7, min(1.5, speed_mult))
        rate = f"{int((speed_mult - 1) * 100):+d}%"

        async def _t():
            c = edge_tts.Communicate(text, vn, rate=rate)
            await c.save(out)

        out = f"/tmp/_tts_{role}_{hash(text) % 10000}.wav"
        asyncio.run(_t())

        if os.path.exists(out):
            with wave.open(out, 'r') as w:
                frames = w.readframes(w.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            os.remove(out)
            return audio
    except:
        pass
    return None


def compose_audio_events(shot, output_path, chat, use_chattts):
    """多轨合成：按 timecode 将 dialogue/VO/SFX/Atmos 混音"""
    events = shot.get("audio_events", {})
    duration = max(shot.get("duration_seconds", 3), 1.0)
    sr = AUDIO_SAMPLE_RATE
    total_samples = int(duration * sr)

    # 限制 duration 在合理范围
    duration = min(duration, 15.0)
    total_samples = int(duration * sr)
    mix = np.zeros(total_samples, dtype=np.float32)

    # 1. Atmos（铺底环境音）
    for evt in events.get("Atmos", []):
        s, e = parse_timecode(evt.get("timecode", "0-1"))
        path = evt.get("path", "")
        vol = float(evt.get("volume", 0.5))
        loop = evt.get("loop", False)

        samples = load_sound_file(path, sr)
        if samples is None:
            continue

        start_sample = int(s * sr)
        end_sample = min(int(e * sr), total_samples)
        if loop and len(samples) < (end_sample - start_sample):
            # 循环铺底
            repeats = (end_sample - start_sample) // len(samples) + 1
            samples = np.tile(samples, repeats)
        seg_len = min(end_sample - start_sample, len(samples))
        if seg_len > 0:
            mix[start_sample:start_sample + seg_len] += samples[:seg_len] * vol

    # 2. SFX（音效）
    for evt in events.get("SFX", []):
        s, e = parse_timecode(evt.get("timecode", "0-1"))
        path = evt.get("path", "")
        vol = float(evt.get("volume", 0.7))

        samples = load_sound_file(path, sr)
        if samples is None:
            continue

        start_sample = int(s * sr)
        end_sample = min(int(e * sr), total_samples)
        seg_len = min(end_sample - start_sample, len(samples))
        if seg_len > 0:
            mix[start_sample:start_sample + seg_len] += samples[:seg_len] * vol

    # 3. Dialogue + VO（按 timecode 嵌入）
    all_voice_events = events.get("dialogue", []) + events.get("VO", [])
    for evt in all_voice_events:
        s, e = parse_timecode(evt.get("timecode", "0-1"))
        text = evt.get("lines", "").strip()
        if not text:
            continue
        role = evt.get("role", "narrator")
        speed_offset = int(evt.get("speed", 0))
        emotion = evt.get("emotion", "calm")
        target_dur = max(e - s, 0.1)

        seg_audio = generate_voice_segment(text, role, speed_offset, emotion, target_dur, chat, use_chattts)
        if seg_audio is None:
            continue

        # 如果语音长度超过目标时长，截断
        target_samples = int(target_dur * sr)
        if len(seg_audio) > target_samples:
            seg_audio = seg_audio[:target_samples]

        start_sample = int(s * sr)
        end_sample = min(start_sample + len(seg_audio), total_samples)
        actual_len = end_sample - start_sample
        if actual_len > 0:
            mix[start_sample:end_sample] += seg_audio[:actual_len]

    # 4. 归一化 + 淡出尾部
    if np.max(np.abs(mix)) > 1.0:
        mix = mix / np.max(np.abs(mix)) * 0.95
    fade_out_samples = min(int(0.3 * sr), len(mix))
    if fade_out_samples > 0:
        mix[-fade_out_samples:] *= np.linspace(1.0, 0.0, fade_out_samples)

    # 5. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wav_int16 = (mix * 32767).astype(np.int16)
    with wave.open(output_path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wav_int16.tobytes())

    return duration


def compose_legacy(shot, output_path, chat, use_chattts):
    """旧模式：纯文本 dialogue/narration → 单轨"""
    char = shot.get("character", "narrator")
    text = shot.get("dialogue") or shot.get("narration") or ""
    emotion = shot.get("emotion", "calm")
    dur = max(shot.get("duration_seconds", 3), 1.0)

    if not text:
        _save_silence(output_path, dur)
        return dur

    voice = VOICE_PARAMS.get(char, VOICE_PARAMS["narrator"]).copy()
    voice["speed"] = voice.get("speed", 1.0) * EMOTION_SPEED.get(emotion, 1.0)

    ok = False

    # ChatTTS
    if chat is not None:
        try:
            wavs = chat.infer([text])
            if wavs and len(wavs) > 0:
                import torchaudio
                audio = wavs[0]
                if isinstance(audio, torch.Tensor):
                    torchaudio.save(output_path, audio.unsqueeze(0), AUDIO_SAMPLE_RATE)
                ok = True
        except:
            pass

    # edge-tts fallback
    if not ok:
        try:
            import asyncio
            vn = {"xiaoming": "zh-CN-YunxiNeural", "xiaoli": "zh-CN-XiaoxiaoNeural",
                  "boss_wang": "zh-CN-YunjianNeural", "narrator": "zh-CN-YunxiNeural"}.get(char, "zh-CN-YunxiNeural")

            async def _t():
                c = edge_tts.Communicate(text, vn)
                await c.save(output_path)

            asyncio.run(_t())
            ok = True
        except:
            pass

    if not ok:
        _save_silence(output_path, dur)

    return dur


def _save_silence(output_path, duration):
    """生成静音 WAV 文件"""
    try:
        sr = AUDIO_SAMPLE_RATE
        n = int(sr * duration)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with wave.open(output_path, 'w') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            for _ in range(n): w.writeframes(struct.pack('<h', 0))
    except Exception:
        subprocess.run(
            f'ffmpeg -y -f lavfi -i "anullsrc=r={AUDIO_SAMPLE_RATE}:cl=mono" '
            f'-t {duration} -acodec pcm_s16le "{output_path}" 2>/dev/null',
            shell=True, timeout=30,
        )


def main():
    parser = argparse.ArgumentParser(description="AI短剧配音生成（多轨合成）")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--output-dir", default="output/audio")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))

    # 尝试加载 ChatTTS
    chat = None
    try:
        import ChatTTS
        # transformers compat patch
        from ChatTTS.model.gpt import GPT
        _orig = GPT._prepare_generation_inputs
        def safe_prepare(self, input_ids, past_key_values=None, attention_mask=None,
                         inputs_embeds=None, cache_position=None, position_ids=None):
            try:
                return _orig(self, input_ids, past_key_values, attention_mask,
                            inputs_embeds, cache_position, position_ids)
            except RuntimeError as e:
                if "narrow" in str(e):
                    import torch
                    if attention_mask is not None and attention_mask.numel() == 0:
                        attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1],
                                                   dtype=torch.long, device=input_ids.device)
                    if cache_position is not None and cache_position.numel() == 0:
                        cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
                    return _orig(self, input_ids, past_key_values, attention_mask,
                                inputs_embeds, cache_position, position_ids)
                raise
        GPT._prepare_generation_inputs = safe_prepare

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        chat = ChatTTS.Chat()
        chat.load(source="local", custom_path=os.path.expanduser("~"))
        log("ChatTTS ✓ (multi-voice)")
    except Exception as e:
        log(f"ChatTTS: {e}")

    # edge-tts 备用
    edge_ok = False
    try:
        import edge_tts
        edge_ok = True
    except:
        pass

    use_chattts = chat is not None
    count = 0

    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = sb.get("episode", 1)
            out = f"{args.output_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.wav"
            os.makedirs(args.output_dir, exist_ok=True)

            if os.path.exists(out) and os.path.getsize(out) > 1000:
                log(f"  [{count}/{total}] {sid} 跳过 (已存在)")
                continue

            # 判断是否有 audio_events（新结构）
            audio_events = shot.get("audio_events", {})
            has_events = audio_events and (
                audio_events.get("dialogue") or
                audio_events.get("VO") or
                audio_events.get("SFX") or
                audio_events.get("Atmos")
            )

            if has_events:
                # 多轨合成模式
                dur = compose_audio_events(shot, out, chat, use_chattts)
                log(f"  [{count}/{total}] {sid} 多轨合成 ✓ ({dur:.1f}s)")
            else:
                # 旧模式：纯文本
                dur = compose_legacy(shot, out, chat, use_chattts)
                log(f"  [{count}/{total}] {sid} ({shot.get('character', '?')}) 旧模式 ✓ ({dur:.1f}s)")

    log("配音生成完成")


if __name__ == "__main__":
    main()
