#!/usr/bin/env python3
"""
Step 4: 视频生成 - Wan2.2 TI2V 5B (直接文本生成视频，跳过 SD 1.5)

使用 ComfyUI API 调用 Wan2.2 TI2V 5B (fp16 safetensors) 直接生成视频。
工作流: UNETLoader → CLIPLoader → VAELoader → ModelSamplingSD3 → Wan22ImageToVideoLatent → KSampler → VAEDecode → VHS_VideoCombine

模型来源: /kaggle/input/saysnkaggle/wan2-2-5b-f16/
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import json
import time
import shutil
import urllib.request
import subprocess


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_wan22_models():
    """查找 Wan2.2 模型文件"""
    search_dirs = [
        WAN22_MODELS_DIR,
        "/kaggle/working/models",
    ]
    result = {"unet": None, "clip": None, "vae": None}
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            fl = f.lower()
            if "wan2.2_ti2v_5b" in fl and "fp16" in fl and f.endswith(".safetensors"):
                result["unet"] = os.path.join(base, f)
            elif "umt5_xxl" in fl and f.endswith(".safetensors"):
                result["clip"] = os.path.join(base, f)
            elif "wan2.2_vae" in fl and f.endswith(".safetensors"):
                result["vae"] = os.path.join(base, f)
    return result


def start_comfyui():
    """启动 ComfyUI"""
    COMFYUI_URL = "http://127.0.0.1:8188"

    # 已在运行？
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
        log("ComfyUI 已在运行")
        return True
    except:
        pass

    # 查找或安装
    comfyui_dir = None
    for c in ["/kaggle/working/ComfyUI", "/kaggle/working/ComfyUI-master"]:
        if os.path.isdir(c) and os.path.isfile(f"{c}/main.py"):
            comfyui_dir = c
            break

    if not comfyui_dir:
        log("安装 ComfyUI...")
        subprocess.run(
            "cd /kaggle/working && git clone https://github.com/comfyanonymous/ComfyUI.git && "
            "cd ComfyUI && pip install -r requirements.txt -q",
            shell=True, timeout=300
        )
        comfyui_dir = "/kaggle/working/ComfyUI"

    # 安装必要的插件
    plugins = {
        "ComfyUI-VideoHelperSuite": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
    }
    for name, url in plugins.items():
        plugin_path = f"{comfyui_dir}/custom_nodes/{name}"
        if not os.path.isdir(plugin_path):
            log(f"安装 {name}...")
            subprocess.run(
                f"cd {comfyui_dir}/custom_nodes && git clone {url}",
                shell=True, timeout=120
            )

    # 创建 extra_model_paths.yaml
    _create_extra_model_paths(comfyui_dir)

    # 启动
    log("启动 ComfyUI...")
    cmd = f"cd {comfyui_dir} && python main.py --listen 0.0.0.0 --dont-print-server 2>&1 &"
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 等待就绪
    for i in range(60):
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
            log(f"ComfyUI 就绪 ({i+1}s)")
            return True
        except:
            time.sleep(2)

    log("ComfyUI 启动超时")
    return False


def _create_extra_model_paths(comfyui_dir):
    """创建 extra_model_paths.yaml 注册模型路径"""
    yaml_content = f"""wan22_ti2v:
  base_path: {WAN22_MODELS_DIR}
  diffusion_models: .
  text_encoders: .
  vae: .
"""
    yaml_path = f"{comfyui_dir}/extra_model_paths.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)


def build_wan22_workflow(positive_prompt, negative_prompt, unet_path, clip_path, vae_path,
                          width=832, height=480, frames=49, steps=20, cfg=5.0,
                          sampler="euler", scheduler="simple", shift=8.0, denoise=1.0,
                          seed=42, fps=8):
    """构建 Wan2.2 TI2V 工作流"""
    # 参考 video_wan2_2_5B_ti2v.json 的架构
    # 使用 UNETLoader (不是 GGUF) + CLIPLoader + VAELoader + ModelSamplingSD3 + Wan22ImageToVideoLatent + KSampler + VAEDecode + VHS_VideoCombine

    unet_name = os.path.basename(unet_path) if unet_path else "wan2.2_ti2v_5B_fp16.safetensors"
    clip_name = os.path.basename(clip_path) if clip_path else "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_name = os.path.basename(vae_path) if vae_path else "wan2.2_vae.safetensors"

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
                "denoise": denoise
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]}
        },
        "10": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["9", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "wan22",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True
            }
        }
    }

    return workflow


def generate_video(workflow, output_path, timeout=1800):
    """通过 ComfyUI API 生成视频"""
    COMFYUI_URL = "http://127.0.0.1:8188"

    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return False
    except Exception as e:
        log(f"  提交失败: {e}")
        return False

    # 等待完成
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            history = json.loads(resp.read())
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {}).get("status_str")
                if status == "success":
                    # 查找视频输出
                    for nid, node_output in entry.get("outputs", {}).items():
                        if "gifs" in node_output:
                            gifs = node_output["gifs"]
                            if gifs and len(gifs) > 0:
                                src = gifs[0]
                                if isinstance(src, dict):
                                    src = src.get("fullpath", "")
                                if src and os.path.isfile(src):
                                    shutil.copy2(src, output_path)
                                    return True
                    log("  完成但无视频输出")
                    return False
                if status == "error":
                    log(f"  执行失败")
                    return False
        except:
            pass
        time.sleep(5)

    log("  超时")
    return False


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V 视频生成")
    parser.add_argument("--storyboard", required=True)
    parser.add_argument("--output-dir", default="output/videos")
    args = parser.parse_args()

    sb = load_json(args.storyboard)
    dirs = get_dirs(sb.get("episode", 1))
    total = sum(len(s.get("shots", [])) for s in sb.get("scenes", []))

    log(f"镜头数: {total}")

    # 查找模型
    models = get_wan22_models()
    log(f"UNET: {os.path.basename(models['unet']) if models['unet'] else '未找到'}")
    log(f"CLIP: {os.path.basename(models['clip']) if models['clip'] else '未找到'}")
    log(f"VAE: {os.path.basename(models['vae']) if models['vae'] else '未找到'}")

    if not models["unet"] or not models["vae"]:
        log("关键模型未找到!")
        sys.exit(1)

    # 启动 ComfyUI
    if not start_comfyui():
        log("ComfyUI 启动失败!")
        sys.exit(1)

    count = 0
    for scene in sb.get("scenes", []):
        for shot in scene.get("shots", []):
            count += 1
            sid = shot["shot_id"]
            ep = sb.get("episode", 1)
            out = f"{args.output_dir}/ep{ep:02d}_{scene['scene_id']}_{sid}.mp4"

            if os.path.exists(out) and os.path.getsize(out) > 100000:
                log(f"[{count}/{total}] {sid} 跳过(已存在)")
                continue

            # 获取 prompt
            prompt = shot.get("prompt", "anime style, high quality")
            neg_prompt = shot.get("negative_prompt", "blurry, distorted, low quality")
            seed = shot.get("seed", 42)
            if isinstance(seed, str):
                seed = 42

            # 调整分辨率
            w = WAN22_WIDTH
            h = WAN22_HEIGHT
            frames = WAN22_FRAMES

            workflow = build_wan22_workflow(
                positive_prompt=prompt,
                negative_prompt=neg_prompt,
                unet_path=models["unet"],
                clip_path=models["clip"],
                vae_path=models["vae"],
                width=w, height=h, frames=frames,
                steps=WAN22_STEPS, cfg=WAN22_CFG,
                sampler=WAN22_SAMPLER, scheduler=WAN22_SCHEDULER,
                shift=WAN22_SHIFT, seed=seed, fps=WAN22_FPS
            )

            log(f"[{count}/{total}] {sid} 生成中 ({w}x{h}, {frames}f)...")
            if generate_video(workflow, out):
                size_mb = os.path.getsize(out) / 1e6
                log(f"[{count}/{total}] {sid} ✓ ({size_mb:.1f}MB)")
            else:
                log(f"[{count}/{total}] {sid} 失败")

    log("视频生成完成")


if __name__ == "__main__":
    main()
