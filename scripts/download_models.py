#!/usr/bin/env python3
"""
下载 Wan2.2 TI2V 5B 模型到 /kaggle/working/models/

模型来源: HF Mirror (hf-mirror.com)
目标目录: /kaggle/working/models/

需要模型:
  1. wan2.2_ti2v_5B_fp16.safetensors (UNET) - ~10GB
  2. umt5_xxl_fp8_e4m3fn_scaled.safetensors (CLIP text encoder) - ~12GB
  3. wan2.2_vae.safetensors (VAE) - ~1.5GB

用法:
  python download_models.py          # 下载所有模型到 /kaggle/working/models/
  python download_models.py --output ./models  # 自定义目录

Kaggle 用法:
  !python download_models.py
  或手动下载后上传到 Dataset。
"""
import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

# ============================================================
# 模型定义 (使用 HF 镜像加速，无需认证)
# ============================================================
HF_MIRROR = "https://hf-mirror.com"

MODELS = {
    "unet": {
        "name": "Wan2.2 TI2V 5B UNET (fp16)",
        "files": [
            {
                "url": f"{HF_MIRROR}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
                "filename": "wan2.2_ti2v_5B_fp16.safetensors",
                "size_gb": 10.0,
                "source": "direct",
            }
        ],
    },
    "clip": {
        "name": "UMT5 XXL Encoder (fp8 scaled)",
        "files": [
            {
                "url": f"{HF_MIRROR}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "size_gb": 12.0,
                "source": "direct",
            }
        ],
    },
    "vae": {
        "name": "Wan2.2 VAE",
        "files": [
            {
                "url": f"{HF_MIRROR}/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors",
                "filename": "wan2.2_vae.safetensors",
                "size_gb": 1.5,
                "source": "direct",
            }
        ],
    },
}


def download_with_aria2(url: str, dest: str) -> bool:
    """用 aria2c 下载（最快，支持断点续传）"""
    try:
        subprocess.run(["aria2c", "--version"], capture_output=True, timeout=5, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    dest_dir = os.path.dirname(dest)
    dest_name = os.path.basename(dest)

    cmd = [
        "aria2c",
        "-x", "8", "-s", "8", "-k", "1M",
        "--async-dns=false",
        "--continue=true",
        f"-d={dest_dir}",
        f"-o={dest_name}",
        url,
    ]

    print(f"    aria2c 下载中...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024


def download_with_wget(url: str, dest: str) -> bool:
    """用 wget 下载"""
    try:
        subprocess.run(["wget", "--version"], capture_output=True, timeout=5, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    cmd = ["wget", "-q", "--show-progress", "--continue", "-O", dest, url]
    print(f"    wget 下载中...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024


def download_file(file_info: dict, output_dir: str) -> str:
    """下载单个文件，自动降级下载方式"""
    url = file_info["url"]
    filename = file_info["filename"]
    expected_gb = file_info["size_gb"]
    dest = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)

    # 检查是否已下载完成
    if os.path.isfile(dest):
        size_gb = os.path.getsize(dest) / (1024 ** 3)
        if size_gb >= expected_gb * 0.9:  # 90% 以上算完成
            print(f"  ✅ 已完成: {filename} ({size_gb:.1f}GB)")
            return dest
        else:
            print(f"  ⚠️ 未完成: {filename} ({size_gb:.2f}GB)，重新下载")
            os.remove(dest)

    print(f"  📥 {filename} (~{expected_gb}GB)")
    print(f"     URL: {url}")

    # 方式1: aria2c
    if download_with_aria2(url, dest):
        size_gb = os.path.getsize(dest) / (1024 ** 3)
        print(f"  ✅ aria2c 完成: {size_gb:.1f}GB")
        return dest

    # 方式2: wget
    if download_with_wget(url, dest):
        size_gb = os.path.getsize(dest) / (1024 ** 3)
        print(f"  ✅ wget 完成: {size_gb:.1f}GB")
        return dest

    # 方式3: 尝试 huggingface_hub (需要 HF_TOKEN)
    try:
        from huggingface_hub import hf_hub_download
        print(f"  ⬇️  huggingface_hub 下载中...")
        repo_id = "/".join(url.split("/")[3:5])
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
        )
        size_gb = os.path.getsize(dest) / (1024 ** 3)
        print(f"  ✅ huggingface_hub 完成: {size_gb:.1f}GB")
        return dest
    except Exception as e:
        print(f"  ❌ 全部方式失败: {e}")
        print(f"  手动下载: {url}")
        print(f"  保存到: {dest}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载 Wan2.2 TI2V 5B 模型")
    parser.add_argument("--output", type=str, default="/kaggle/working/models",
                        help="模型输出目录 (默认: /kaggle/working/models)")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()), help="要下载的模型")
    args = parser.parse_args()

    output_dir = args.output
    print(f"📁 模型目录: {output_dir}")
    print()

    results = {}
    for model_key in args.models:
        model_info = MODELS[model_key]
        print(f"{'=' * 50}")
        print(f"📦 {model_info['name']}")
        print(f"{'=' * 50}")

        for file_info in model_info["files"]:
            dest = download_file(file_info, output_dir)
            results[file_info["filename"]] = dest
            print()

    # 总结
    print("=" * 50)
    print("📊 下载结果:")
    print("=" * 50)
    all_ok = True
    for filename, path in results.items():
        if path:
            size_gb = os.path.getsize(path) / (1024 ** 3)
            print(f"  ✅ {filename}: {size_gb:.1f}GB")
        else:
            print(f"  ❌ {filename}: 下载失败")
            all_ok = False

    if all_ok:
        print(f"\n✅ 所有模型已下载到: {output_dir}")
    else:
        print(f"\n⚠️ 部分模型下载失败，请检查网络或手动下载")


if __name__ == "__main__":
    main()
