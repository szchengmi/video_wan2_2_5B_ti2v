#!/usr/bin/env python3
"""
一键下载 Wan2.2 TI2V 5B 模型 — 直接 wget，不依赖 huggingface_hub
目标: /kaggle/working/models/

用法:
  !rm -rf /kaggle/working/models && python download_models.py

不需要 HF_TOKEN，直接通过 HF 镜像下载。
"""
import os
import time
import subprocess
import shutil

MODELS = [
    {
        "name": "Wan2.2 TI2V 5B UNET",
        "dir": "wan22_ti2v_5b",
        "files": [
            "https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        ],
    },
    {
        "name": "Wan2.2 VAE",
        "dir": "wan22_vae",
        "files": [
            "https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
        ],
    },
    {
        "name": "UMT5 XXL Encoder",
        "dir": "umt5_xxl",
        "files": [
            "https://hf-mirror.com/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ],
    },
]

BASE = "/kaggle/working/models"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_via_wget(url, dest_path):
    """wget 下载，带进度"""
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    # 已存在且足够大
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 100 * 1024 * 1024:
        size_mb = os.path.getsize(dest_path) / 1e6
        log(f"  ✅ 已存在 ({size_mb:.0f}MB)")
        return True

    log(f"  ⬇️  {os.path.basename(dest_path)}...")
    log(f"      {url}")

    cmd = [
        "wget",
        "--progress=dot:gigabytes",
        "--no-check-certificate",
        "-O", dest_path,
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode == 0 and os.path.isfile(dest_path) and os.path.getsize(dest_path) > 100 * 1024 * 1024:
        size_mb = os.path.getsize(dest_path) / 1e6
        log(f"  ✅ 完成: {size_mb:.0f}MB")
        return True
    else:
        log(f"  ❌ wget 失败 (rc={result.returncode})")
        if os.path.isfile(dest_path):
            os.remove(dest_path)
        return False


def main():
    log("=" * 55)
    log("  Wan2.2 TI2V 5B 模型下载 (HF Mirror)")
    log("=" * 55)
    log(f"目标: {BASE}")

    # 清理
    if os.path.isdir(BASE):
        shutil.rmtree(BASE)
    os.makedirs(BASE, exist_ok=True)

    ok_total = 0
    fail_total = 0

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']}")
        target = f"{BASE}/{model['dir']}"
        os.makedirs(target, exist_ok=True)

        for url in model["files"]:
            filename = url.split("/")[-1]
            dest = f"{target}/{filename}"
            if download_via_wget(url, dest):
                ok_total += 1
            else:
                fail_total += 1

    log(f"\n{'='*55}")
    log(f"完成！成功: {ok_total}, 失败: {fail_total}")
    for model in MODELS:
        path = f"{BASE}/{model['dir']}"
        size = sum(os.path.getsize(f"{path}/{f}") for f in os.listdir(path) if os.path.isfile(f"{path}/{f}")) / 1e9 if os.path.isdir(path) else 0
        log(f"  {model['name']}: {size:.2f}GB")


if __name__ == "__main__":
    main()
