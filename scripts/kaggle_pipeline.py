#!/usr/bin/env python3
"""
video_wan2_2_5B_ti2v — AI短剧自动生成 Pipeline (Kaggle T4 优化版)

基于本地工作流 video_wan2_2_5B_ti2v.json 构建。
使用 Wan2.2 TI2V 5B (fp16 safetensors) 直接文本生成视频。

架构:
  Step 1: Qwen2.5-3B 生成剧本 (LLM)
  Step 2: 分镜解析
  Step 3: ComfyUI + Wan2.2 TI2V 生成视频片段
  Step 4: edge-tts 配音
  Step 5: FFmpeg 合成最终视频

Kaggle 运行:
  !git clone https://github.com/szchengmi/video_wan2_2_5B_ti2v.git
  !cd video_wan2_2_5B_ti2v/scripts && python kaggle_pipeline.py --force

模型准备:
  1. 运行: python download_models.py
  2. 或 Kaggle Dataset 挂载到 /kaggle/input/your-dataset
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import re
import gc
from pathlib import Path

# ============================================================
# 配置
# ============================================================
MODELS_DIR = "/kaggle/working/models"
OUTPUT_DIR = "/kaggle/working/ai-series"

# 视频参数 (Wan2.2 TI2V 5B 推荐)
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_FRAMES = 49   # ~6s @ 8fps, 需满足 (length-1)%4==0
DEFAULT_FPS = 8
DEFAULT_STEPS = 20
DEFAULT_CFG = 5.0
DEFAULT_SAMPLER = "euler"
DEFAULT_SCHEDULER = "simple"
DEFAULT_SHIFT = 8.0
DEFAULT_SEED = 42

# 剧本参数
EPISODE_NUM = 1
NUM_SCENES = 3
SHOTS_PER_SCENE = 2


# ============================================================
# 工具函数
# ============================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_cmd(cmd, timeout=300):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 and result.stderr:
        log(f"  stderr: {result.stderr[:200]}")
    return result


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def detect_environment():
    """检测运行环境"""
    import torch
    is_kaggle = os.path.isdir("/kaggle/input")
    has_gpu = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_gpu else "N/A"
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9 if has_gpu else 0
    import psutil
    total_ram = psutil.virtual_memory().total / 1e9
    cpu_count = psutil.cpu_count()
    return {
        "is_kaggle": is_kaggle,
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "gpu_mem_gb": gpu_mem,
        "total_memory_gb": total_ram,
        "cpu_count": cpu_count,
    }


def find_models():
    """查找模型文件"""
    search_dirs = [
        MODELS_DIR,
        "/kaggle/working/models",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"),
    ]
    result = {"unet": None, "clip": None, "vae": None}
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            for f in files:
                if "wan2.2_ti2v_5b" in f.lower() and "fp16" in f.lower() and f.endswith(".safetensors"):
                    result["unet"] = os.path.join(root, f)
                elif "umt5_xxl" in f.lower() and f.endswith(".safetensors"):
                    result["clip"] = os.path.join(root, f)
                elif "wan2.2_vae" in f.lower() and f.endswith(".safetensors"):
                    result["vae"] = os.path.join(root, f)
    return result


def get_dirs(ep):
    """获取输出目录结构"""
    base = OUTPUT_DIR
    dirs = {
        "script": f"{base}/episode_{ep:02d}",
        "storyboard": f"{base}/episode_{ep:02d}",
        "videos": f"{base}/episode_{ep:02d}/videos",
        "audio": f"{base}/episode_{ep:02d}/audio",
        "final": f"{base}/episode_{ep:02d}/final",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# Step 1: 剧本生成 (Qwen2.5-3B)
# ============================================================
def step1_generate_script(force=False):
    """通过本地 Qwen2.5-3B 生成短剧剧本"""
    dirs = get_dirs(EPISODE_NUM)
    out_path = f"{dirs['script']}/episode_{EPISODE_NUM:02d}_script.json"

    if not force and os.path.exists(out_path):
        log(f"  跳过(已存在): {out_path}")
        return load_json(out_path)

    # 查找 Qwen 模型
    qwen_paths = [
        "/kaggle/input/qwen25-3b-instruct",
        "/kaggle/working/qwen",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "qwen"),
    ]
    qwen_model = None
    for p in qwen_paths:
        if os.path.isdir(p) and os.path.isfile(f"{p}/config.json"):
            qwen_model = p
            break

    if not qwen_model:
        log("  ❌ Qwen 模型未找到，使用内置 demo 剧本")
        return _generate_demo_script(out_path)

    log(f"  Qwen: {qwen_model}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(qwen_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    prompt = _build_script_prompt()
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    log("  生成剧本...")
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(generated[0], skip_special_tokens=True)
    # 提取 response 部分
    if "Response" in raw:
        raw = raw.split("Response")[-1].strip()
    elif "response" in raw:
        raw = raw.split("response")[-1].strip()

    # 保存原始输出
    raw_path = "/kaggle/working/ai-series/qwen_raw_output.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw)

    # 卸载 Qwen 释放 GPU
    log("  卸载 Qwen 释放 GPU...")
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 解析剧本
    script = _parse_script(raw, out_path)
    return script


def _build_script_prompt():
    return f"""你是一位专业的短视频编剧。请创作一个都市爱情短剧剧本。

要求:
- 共 {NUM_SCENES} 个场景，每场景 {SHOTS_PER_SCENE} 个镜头
- 总时长 3-5 分钟
- 角色: 男主角(程序员)，女主角(设计师)
- 故事: 两人在地铁相遇，经历误会最终相识

请用以下 JSON 格式返回:
{{
  "title": "剧名",
  "total_duration": "总时长(如3分30秒)",
  "estimated_time": "3-5分钟",
  "characters": [
    {{"name": "名字", "role": "角色", "gender": "性别", "age": "年龄", "appearance": "外貌特征描述(英文)"}}
  ],
  "scenes": [
    {{
      "scene_id": "scene_1",
      "setting": "场景描述",
      "duration": "场景时长",
      "shots": [
        {{
          "shot_id": "scene_1_shot_1",
          "duration_seconds": 5,
          "camera": "镜头描述",
          "action": "动作描述",
          "dialogue": "台词",
          "prompt": "English visual description for video generation, detailed, anime style"
        }}
      ]
    }}
  ]
}}

只返回 JSON，不要其他内容。"""


def _parse_script(raw, out_path):
    """解析 Qwen 输出的 JSON"""
    # 提取 JSON 部分
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    # 修复常见 JSON 问题
    text = text.replace("\\n", "\n").replace("\\r", "\r")
    text = re.sub(r'"\s*\n\s*"', '",\n"', text)
    text = re.sub(r'"\s+"', ', "', text)
    text = re.sub(r'(?<=[\\]}])\s+"', ', "', text)
    text = re.sub(r',(\s*[\\]}])', r'\1', text)

    try:
        data = json.loads(text)
        save_json(out_path, data)
        log(f"  ✅ 剧本解析成功: {len(data.get('scenes', []))} 场景")
        return data
    except json.JSONDecodeError:
        # 尝试截取第一个完整 JSON
        depth = 0
        start = -1
        for i, c in enumerate(text):
            if c == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        data = json.loads(text[start:i + 1])
                        save_json(out_path, data)
                        log(f"  ✅ 截取 JSON 成功")
                        return data
                    except:
                        continue
        log("  ❌ JSON 解析失败，使用 demo 剧本")
        return _generate_demo_script(out_path)


def _generate_demo_script(out_path):
    """生成 demo 剧本（fallback）"""
    data = {
        "title": "地铁邂逅",
        "total_duration": "3分钟",
        "estimated_time": "3分钟",
        "characters": [
            {"name": "李明", "role": "男主角", "gender": "male", "age": "28", "appearance": "young Chinese man, short black hair, glasses, wearing dark hoodie"},
            {"name": "苏晴", "role": "女主角", "gender": "female", "age": "26", "appearance": "young Chinese woman, long black hair, wearing light sweater"}
        ],
        "scenes": []
    }

    scenes_data = [
        {
            "scene_id": "scene_1",
            "setting": "地铁站 - 早高峰",
            "duration": "6秒",
            "shots": [
                {"shot_id": "scene_1_shot_1", "duration_seconds": 3, "camera": "wide shot", "action": "人群走进地铁站", "dialogue": "", "prompt": "crowded subway station morning rush hour, people walking, warm lighting, anime style, high quality"},
                {"shot_id": "scene_1_shot_2", "duration_seconds": 3, "camera": "medium shot", "action": "男主角在看手机", "dialogue": "", "prompt": "young man checking phone in subway station, wearing dark hoodie, anime style, high quality"},
            ]
        },
        {
            "scene_id": "scene_2",
            "setting": "地铁车厢内",
            "duration": "6秒",
            "shots": [
                {"shot_id": "scene_2_shot_1", "duration_seconds": 3, "camera": "medium shot", "action": "女主角在看书", "dialogue": "", "prompt": "young woman reading book in subway car, long black hair, soft lighting, anime style, high quality"},
                {"shot_id": "scene_2_shot_2", "duration_seconds": 3, "camera": "close up", "action": "男主角偷偷看她", "dialogue": "", "prompt": "young man secretly glancing at woman across subway car, anime style, high quality"},
            ]
        },
        {
            "scene_id": "scene_3",
            "setting": "地铁站出口",
            "duration": "6秒",
            "shots": [
                {"shot_id": "scene_3_shot_1", "duration_seconds": 3, "camera": "medium shot", "action": "两人同时走出地铁", "dialogue": "", "prompt": "man and woman walking out of subway station together, sunset light, anime style, high quality"},
                {"shot_id": "scene_3_shot_2", "duration_seconds": 3, "camera": "wide shot", "action": "女主角回头微笑", "dialogue": "", "prompt": "woman turning head and smiling at man on street, warm sunset, anime style, high quality"},
            ]
        }
    ]

    data["scenes"] = scenes_data
    save_json(out_path, data)
    return data


# ============================================================
# Step 2: 分镜生成
# ============================================================
def step2_generate_storyboard(script_data, force=False):
    """将剧本转换为视频分镜"""
    dirs = get_dirs(EPISODE_NUM)
    out_path = f"{dirs['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"

    if not force and os.path.exists(out_path):
        log(f"  跳过(已存在): {out_path}")
        return load_json(out_path)

    # 分镜直接使用剧本中的 shots，添加 seed
    import random
    storyboard = {
        "episode": EPISODE_NUM,
        "title": script_data.get("title", "AI短剧"),
        "characters": script_data.get("characters", []),
        "scenes": []
    }

    for scene in script_data.get("scenes", []):
        new_scene = {
            "scene_id": scene["scene_id"],
            "setting": scene.get("setting", ""),
            "shots": []
        }
        for shot in scene.get("shots", []):
            new_shot = dict(shot)
            new_shot["seed"] = random.randint(1, 999999)
            new_scene["shots"].append(new_shot)
        storyboard["scenes"].append(new_scene)

    save_json(out_path, storyboard)
    total = sum(len(s.get("shots", [])) for s in storyboard["scenes"])
    log(f"  分镜完成: {len(storyboard['scenes'])} 场景, {total} 镜头")
    return storyboard


# ============================================================
# Step 3: 视频生成 (Wan2.2 TI2V 5B via ComfyUI)
# ============================================================
def step3_generate_videos(storyboard, force=False):
    """通过 ComfyUI API 调用 Wan2.2 TI2V 生成视频"""
    dirs = get_dirs(EPISODE_NUM)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"  镜头数: {total}")

    # 查找模型
    models = find_models()
    for name, path in models.items():
        if path:
            log(f"  ✓ {name}: {os.path.basename(path)}")
        else:
            log(f"  ✗ {name}: 未找到")

    if not models["unet"] or not models["vae"]:
        log("  ❌ UNET 或 VAE 未找到!")
        return

    # 启动 ComfyUI
    if not _start_comfyui():
        log("  ❌ ComfyUI 启动失败")
        return

    # 构建并执行工作流
    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = storyboard.get("episode", 1)
            out = f"{dirs['videos']}/ep{ep:02d}_{sid}.mp4"

            if os.path.exists(out) and os.path.getsize(out) > 100000:
                log(f"  [{count}/{total}] {sid} 跳过(已存在)")
                continue

            video_prompt = shot.get("prompt", "anime style, high quality")
            neg_prompt = "blurry, distorted, low quality, static, motionless, low contrast"
            seed = shot.get("seed", DEFAULT_SEED)

            try:
                _generate_single_video(
                    positive_prompt=video_prompt,
                    negative_prompt=neg_prompt,
                    unet_path=models["unet"],
                    clip_path=models["clip"],
                    vae_path=models["vae"],
                    width=DEFAULT_WIDTH,
                    height=DEFAULT_HEIGHT,
                    frames=DEFAULT_FRAMES,
                    fps=DEFAULT_FPS,
                    steps=DEFAULT_STEPS,
                    cfg=DEFAULT_CFG,
                    sampler=DEFAULT_SAMPLER,
                    scheduler=DEFAULT_SCHEDULER,
                    shift=DEFAULT_SHIFT,
                    seed=seed,
                    output_path=out,
                    count=count,
                    total=total,
                    sid=sid,
                )
            except Exception as e:
                log(f"  [{count}/{total}] {sid} 失败: {e}")
                _save_placeholder_video(out)

    log("  视频生成完成")


def _generate_single_video(positive_prompt, negative_prompt, unet_path, clip_path, vae_path,
                           width, height, frames, fps, steps, cfg, sampler, scheduler,
                           shift, seed, output_path, count, total, sid):
    """生成单个视频片段"""
    import urllib.request

    COMFYUI_URL = "http://127.0.0.1:8188"

    # 构建工作流 (参考 video_wan2_2_5B_ti2v.json 架构)
    # 节点映射:
    #   1: UNETLoader
    #   2: CLIPLoader
    #   3: VAELoader
    #   4: CLIPTextEncode (positive)
    #   5: CLIPTextEncode (negative)
    #   6: ModelSamplingSD3
    #   7: Wan22ImageToVideoLatent
    #   8: KSampler
    #   9: VAEDecode
    #   10: CreateVideo
    #   11: SaveVideo

    unet_name = os.path.basename(unet_path)
    clip_name = os.path.basename(clip_path) if clip_path else "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_name = os.path.basename(vae_path)

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_name, "weight_dtype": "default"}
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "wan", "device": "default"}
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name}
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["2", 0]}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]}
        },
        "6": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": ["1", 0], "shift": shift}
        },
        "7": {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["3", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]}
        },
        "10": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["9", 0], "fps": fps}
        },
        "11": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["10", 0],
                "filename_prefix": f"ep{count:02d}_{sid}",
                "format": "auto",
                "codec": "auto"
            }
        }
    }

    # 提交 prompt
    log(f"  [{count}/{total}] {sid} 提交中...")
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    result = json.loads(resp.read())
    prompt_id = result.get("prompt_id")

    if not prompt_id:
        raise RuntimeError(f"API 返回无 prompt_id")

    log(f"  [{count}/{total}] {sid} 等待完成 (id={prompt_id[:8]}...)")

    # 等待完成
    start = time.time()
    while time.time() - start < 1800:  # 30分钟超时
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            history = json.loads(resp.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {}).get("status_str")
                if status == "success":
                    # 查找视频输出
                    for nid, node_output in entry.get("outputs", {}).items():
                        if "video" in node_output:
                            video_info = node_output["video"]
                            if isinstance(video_info, list) and len(video_info) > 0:
                                src = video_info[0]
                                if isinstance(src, dict):
                                    src = src.get("fullpath", src.get("filename", ""))
                                if src and os.path.isfile(src):
                                    shutil.copy2(src, output_path)
                                    size_mb = os.path.getsize(output_path) / 1e6
                                    log(f"  [{count}/{total}] {sid} ✓ ({size_mb:.1f}MB)")
                                    return
                        if "vhs_filenames" in node_output:
                            src = node_output["vhs_filenames"][0]
                            if isinstance(src, dict):
                                src = src.get("fullpath", "")
                            if src and os.path.isfile(src):
                                shutil.copy2(src, output_path)
                                size_mb = os.path.getsize(output_path) / 1e6
                                log(f"  [{count}/{total}] {sid} ✓ ({size_mb:.1f}MB)")
                                return
                    log(f"  [{count}/{total}] {sid} 无输出视频")
                    return
                if status == "error":
                    raise RuntimeError(f"工作流执行失败")
        except urllib.error.HTTPError:
            pass
        time.sleep(5)

    raise TimeoutError(f"超时")


def _save_placeholder_video(output_path):
    """保存占位视频"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_cmd(
        f'ffmpeg -y -f lavfi -i color=c=1a1a2e:s=320x240:d=3 '
        f'-c:v libx264 -pix_fmt yuv420p "{output_path}" 2>/dev/null'
    )


# ============================================================
# ComfyUI 管理
# ============================================================
def _start_comfyui():
    """启动 ComfyUI 服务器"""
    import urllib.request
    COMFYUI_URL = "http://127.0.0.1:8188"

    # 已在运行？
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        log("  ComfyUI 已在运行")
        return True
    except:
        pass

    # 查找或安装 ComfyUI
    comfyui_dir = _find_or_install_comfyui()
    if not comfyui_dir:
        return False

    # 创建 extra_model_paths.yaml
    _create_extra_model_paths(comfyui_dir)

    # 启动
    log("  启动 ComfyUI...")
    cmd = f"cd {comfyui_dir} && python main.py --listen 0.0.0.0 --dont-print-server"
    if detect_environment()["has_gpu"]:
        cmd += " --cuda-device 0"

    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 等待就绪
    for i in range(60):
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
            log(f"  ComfyUI 就绪 ({i+1}s)")
            return True
        except:
            time.sleep(2)

    log("  ❌ ComfyUI 启动超时")
    return False


def _find_or_install_comfyui():
    """查找或安装 ComfyUI"""
    # 检查已有安装
    for candidate in ["/kaggle/working/ComfyUI", "/kaggle/working/ComfyUI-master"]:
        if os.path.isdir(candidate) and os.path.isfile(f"{candidate}/main.py"):
            return candidate

    # 安装
    log("  安装 ComfyUI...")
    run_cmd(
        "cd /kaggle/working && "
        "git clone https://github.com/comfyanonymous/ComfyUI.git && "
        "cd ComfyUI && pip install -r requirements.txt -q",
        timeout=300
    )

    comfyui_dir = "/kaggle/working/ComfyUI"
    if os.path.isfile(f"{comfyui_dir}/main.py"):
        return comfyui_dir
    return None


def _create_extra_model_paths(comfyui_dir):
    """创建 extra_model_paths.yaml 注册模型路径"""
    models_base = MODELS_DIR

    yaml_content = f"""wan22_ti2v:
  base_path: {models_base}
  diffusion_models: .
  text_encoders: .
  vae: .
"""

    yaml_path = f"{comfyui_dir}/extra_model_paths.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    log(f"  创建 extra_model_paths.yaml: {yaml_path}")


# ============================================================
# Step 4: 配音生成
# ============================================================
def step4_generate_audio(storyboard, force=False):
    """生成配音"""
    dirs = get_dirs(EPISODE_NUM)

    try:
        import edge_tts
    except ImportError:
        log("  安装 edge-tts...")
        run_cmd("pip install -q edge-tts", timeout=60)
        import edge_tts

    async def _gen():
        communicate = edge_tts.Communicate("zh-CN-XiaoxiaoxiaoNeural", "Hello, this is a test.", rate="-10%")
        await communicate.save("/tmp/test_tts.mp3")

    import asyncio
    asyncio.run(_gen())

    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    count = 0
    for scene in storyboard.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            out = f"{dirs['audio']}/ep{EPISODE_NUM:02d}_{sid}.mp3"
            dialogue = shot.get("dialogue", "")

            if not dialogue:
                # 无台词生成静音
                run_cmd(f'ffmpeg -y -f lavfi -i anullsrc=r=22050:cl=mono:d=3 -c:a libmp3lame "{out}" 2>/dev/null')
                continue

            try:
                communicate = edge_tts.Communicate("zh-CN-XiaoxiaoxiaoNeural", dialogue, rate="-10%")
                asyncio.run(communicate.save(out))
                log(f"  [{count}/{total}] {sid} 配音完成")
            except Exception as e:
                log(f"  [{count}/{total}] {sid} 配音失败: {e}")

    log("  配音生成完成")


# ============================================================
# Step 5: 剪辑合成
# ============================================================
def step5_compose(storyboard, script_data=None):
    """最终合成"""
    dirs = get_dirs(EPISODE_NUM)
    videos_dir = dirs["videos"]
    audio_dir = dirs["audio"]
    out = f"{dirs['final']}/episode_{EPISODE_NUM:02d}_final.mp4"

    # 收集视频文件
    video_files = sorted(glob.glob(f"{videos_dir}/ep{EPISODE_NUM:02d}_*.mp4"))
    if not video_files:
        log("  ❌ 无视频文件可合成")
        return None

    log(f"  合成 {len(video_files)} 个视频片段...")

    # 创建 concat 文件
    concat_file = "/tmp/ffmpeg_concat.txt"
    with open(concat_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")

    # concat demuxer 合成
    run_cmd(
        f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" '
        f'-c:v libx264 -pix_fmt yuv420p -movflags +faststart "{out}" 2>/dev/null',
        timeout=120
    )

    if os.path.exists(out) and os.path.getsize(out) > 100000:
        size_mb = os.path.getsize(out) / 1e6
        dur = _get_video_duration(out)
        log(f"  ✅ 合成完成: {out} ({size_mb:.1f}MB, {dur:.1f}s)")
        return out
    else:
        log("  ❌ 合成失败")
        return None


def _get_video_duration(path):
    """获取视频时长"""
    try:
        result = run_cmd(
            f'ffprobe -v quiet -show_entries format=duration -of csv=p=0 "{path}"'
        )
        return float(result.stdout.strip())
    except:
        return 0


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="video_wan2_2_5B_ti2v AI短剧生成")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    import common
    if args.episode is not None:
        global EPISODE_NUM
        EPISODE_NUM = args.episode

    log("╔══════════════════════════════════════════╗")
    log("║   AI短剧自动生成 — Wan2.2 TI2V 5B        ║")
    log("╚══════════════════════════════════════════╝")

    # 环境检测
    env = detect_environment()
    log(f"环境: {'Kaggle' if env['is_kaggle'] else '本地'} | "
        f"GPU: {env['gpu_name']} ({env['gpu_mem_gb']:.1f}GB) | "
        f"RAM: {env['total_memory_gb']:.1f}GB")

    # 模型检测
    models = find_models()
    log("模型检测:")
    for name, path in models.items():
        if path:
            log(f"  ✓ {name}: {os.path.basename(path)}")
        else:
            log(f"  ✗ {name}: 未找到")

    if not models["unet"] or not models["vae"]:
        log("❌ 关键模型未找到!")
        log("请先运行: python download_models.py")
        log("或在 Kaggle 中挂载包含模型的 Dataset")
        return

    # 安装依赖
    log("\n安装依赖...")
    run_cmd("pip install -q edge-tts psutil", timeout=60)

    # 清除旧输出
    if args.force:
        log("清除旧输出...")
        if os.path.isdir(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

    start_time = time.time()

    # Step 1
    log("\n" + "=" * 50)
    log("Step 1: 剧本生成")
    log("=" * 50)
    t = time.time()
    script_data = step1_generate_script(force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    # Step 2
    log("\n" + "=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)
    t = time.time()
    storyboard = step2_generate_storyboard(script_data, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    # Step 3
    log("\n" + "=" * 50)
    log("Step 3: 视频生成 (Wan2.2 TI2V)")
    log("=" * 50)
    t = time.time()
    step3_generate_videos(storyboard, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    # Step 4
    log("\n" + "=" * 50)
    log("Step 4: 配音生成")
    log("=" * 50)
    t = time.time()
    step4_generate_audio(storyboard, force=args.force)
    log(f"  耗时: {time.time() - t:.1f}s")

    # Step 5
    log("\n" + "=" * 50)
    log("Step 5: 剪辑合成")
    log("=" * 50)
    t = time.time()
    final = step5_compose(storyboard, script_data)
    log(f"  耗时: {time.time() - t:.1f}s")

    # 总结
    total_time = time.time() - start_time
    log("\n" + "=" * 50)
    log("完成!")
    log("=" * 50)
    log(f"总耗时: {total_time:.0f}s ({total_time / 60:.1f}min)")
    if final:
        log(f"输出: {final}")


if __name__ == "__main__":
    import glob
    main()
