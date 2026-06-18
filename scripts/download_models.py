#!/usr/bin/env python3
"""
下载 Wan2.2 TI2V 5B 模型到本地目录。
模型来源: HuggingFace
目标目录: /Users/heipi/heipiworkspace/mac/projects/video_wan2_2_5B_ti2v/models/

需要模型:
  1. wan2.2_ti2v_5B_fp16.safetensors (UNET) - ~10GB
  2. umt5_xxl_fp8_e4m3fn_scaled.safetensors (CLIP text encoder) - ~12GB
  3. wan2.2_vae.safetensors (VAE) - ~1.5GB

用法:
  python download_models.py          # 下载所有模型
  python download_models.py --models unet  # 只下载 UNET
  python download_models.py --models clip  # 只下载 CLIP
  python download_models.py --models vae   # 只下载 VAE

Kaggle 用法 (先挂载 Dataset 到 /kaggle/input):
  !python download_models.py --output /kaggle/working/models
"""
import os
import re
import sys
import argparse
from pathlib import Path

# ============================================================
# 模型定义
# ============================================================
MODELS = {
    "unet": {
        "name": "Wan2.2 TI2V 5B UNET (fp16)",
        "files": [
            {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_RAW/resolve/main/wan2.2_ti2v_5B_fp16.safetensors",
                "filename": "wan2.2_ti2v_5B_fp16.safetensors",
                "size_gb": 10.0,
                "sha256": None,
            }
        ],
    },
    "clip": {
        "name": "UMT5 XXL Encoder (fp8 scaled)",
        "files": [
            {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_RAW/resolve/main/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "size_gb": 12.0,
                "sha256": None,
            }
        ],
    },
    "vae": {
        "name": "Wan2.2 VAE",
        "files": [
            {
                "url": "https://huggingface.co/Comfy-Org/Wan_2.2_RAW/resolve/main/wan2.2_vae.safetensors",
                "filename": "wan2.2_vae.safetensors",
                "size_gb": 1.5,
                "sha256": None,
            }
        ],
    },
}

# HuggingFace 镜像 (中国大陆加速)
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


def download_file(url: str, dest: str, desc: str = "") -> str:
    """下载文件，自动尝试镜像，支持断点续传"""
    import subprocess

    dest = os.path.abspath(dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:  # >100MB
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  ✅ 已存在: {dest} ({size_mb:.0f}MB)")
        return dest

    # 尝试 aria2c (最快)
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, timeout=5, check=True)
        for mirror in HF_MIRRORS:
            mirror_url = url.replace("https://huggingface.co", mirror)
            cmd = [
                "aria2c", "-x", "8", "-s", "8", "-k", "1M",
                "--async-dns=false",
                f"-d={os.path.dirname(dest)}",
                f"-o={os.path.basename(dest)}",
                mirror_url,
            ]
            print(f"  ⬇️  aria2c 下载 {desc}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  ✅ 下载完成: {dest} ({size_mb:.0f}MB)")
                return dest
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 尝试 wget
    try:
        for mirror in HF_MIRRORS:
            mirror_url = url.replace("https://huggingface.co", mirror)
            cmd = [
                "wget", "-q", "--show-progress",
                "-O", dest,
                mirror_url,
            ]
            print(f"  ⬇️  wget 下载 {desc}...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  ✅ 下载完成: {dest} ({size_mb:.0f}MB)")
                return dest
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # 最后尝试 huggingface_hub
    try:
        from huggingface_hub import hf_hub_download
        print(f"  ⬇️  huggingface_hub 下载 {desc}...")
        repo_id = "/".join(url.split("/")[3:5])  # Comfy-Org/Wan_2.2_RAW
        filename = url.split("/")[-1]
        dest = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(dest),
        )
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  ✅ 下载完成: {dest} ({size_mb:.0f}MB)")
        return dest
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        print(f"  手动下载: {url}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载 Wan2.2 TI2V 5B 模型")
    parser.add_argument("--output", type=str, default=None, help="模型输出目录 (默认: ./models)")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()), help="要下载的模型")
    args = parser.parse_args()

    # 确定输出目录
    if args.output:
        output_dir = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), "models")

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 模型目录: {output_dir}")
    print()

    for model_key in args.models:
        model_info = MODELS[model_key]
        print(f"{'=' * 50}")
        print(f"📦 {model_info['name']}")
        print(f"{'=' * 50}")

        for file_info in model_info["files"]:
            url = file_info["url"]
            filename = file_info["filename"]
            dest = os.path.join(output_dir, filename)

            print(f"  📄 {filename} (~{file_info['size_gb']}GB)")
            download_file(url, dest, desc=filename)
            print()

    print("✅ 所有模型下载完成!")
    print(f"   目录: {output_dir}")
    print()
    print("Kaggle 使用方式:")
    print("  1. 创建 Dataset，上传这 3 个模型文件")
    print("  2. 在 Notebook 中挂载 Dataset 到 /kaggle/input/your-dataset-name")
    print("  3. 运行 pipeline 时指定 --models-path /kaggle/input/your-dataset-name")


if __name__ == "__main__":
    main()
