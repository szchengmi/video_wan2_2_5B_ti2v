#!/usr/bin/env python3
"""
Wan2.2 TI2V 5B 模型下载
====================
直接指定文件名，用 huggingface_hub 下载。

Kaggle 用法:
  1. 添加 Secret: HF_TOKEN (你的 HuggingFace token)
  2. !python download_models.py

模型:
  1. wan2.2_ti2v_5B_fp16.safetensors (UNET) - ~10GB
  2. umt5_xxl_fp8_e4m3fn_scaled.safetensors (CLIP) - ~12GB
  3. wan2.2_vae.safetensors (VAE) - ~1.5GB
"""

import os
import sys
import time
import shutil
import subprocess

MODEL_CACHE_DIR = "/kaggle/working/models"


def get_kaggle_secret(key_name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass
    return os.environ.get(key_name, "")


HF_TOKEN = get_kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print(f"[OK] HF_TOKEN: {HF_TOKEN[:10]}...")
else:
    print("[WARN] HF_TOKEN 未设置 (Kaggle Secrets 或环境变量)")


os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def get_dir_size_gb(path):
    total = 0
    if not os.path.exists(path):
        return 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
            except:
                pass
    return total / 1e9


def download_file(model_id, filename, dest_dir):
    """下载单个文件，保存为 bare filename"""
    from huggingface_hub import hf_hub_download

    bare_name = os.path.basename(filename)
    dest = f"{dest_dir}/{bare_name}"

    # 已存在且足够大
    if os.path.isfile(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        size_mb = os.path.getsize(dest) / 1e6
        log(f"  ✅ {bare_name} (已存在, {size_mb:.0f}MB)")
        return True

    os.makedirs(dest_dir, exist_ok=True)

    try:
        # hf_hub_download 保存时会保留子目录结构
        # 下载后需要移动到 dest_dir 根目录
        result_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=dest_dir,
        )
        # 如果保存路径包含子目录，移动到根目录
        if result_path != dest and os.path.isfile(result_path):
            os.rename(result_path, dest)
        size_mb = os.path.getsize(dest) / 1e6
        log(f"  ✅ {bare_name} ({size_mb:.0f}MB)")
        return True
    except Exception as e:
        log(f"  ❌ {bare_name}: {e}")
        return False


# ============================================================
# 模型定义
# ============================================================

MODELS = [
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 TI2V 5B",
        "dir": "wan22_ti2v_5b",
        "desc": "~10GB",
        "files": [
            "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "name": "UMT5 XXL Encoder",
        "dir": "umt5_xxl",
        "desc": "~12GB",
        "files": [
            "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 VAE",
        "dir": "wan22_vae",
        "desc": "~1.5GB",
        "files": [
            "split_files/vae/wan2.2_vae.safetensors",
        ],
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Wan2.2 TI2V 5B 模型下载")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")

    # 清理旧文件
    if os.path.isdir(MODEL_CACHE_DIR):
        log(f"清理旧目录: {MODEL_CACHE_DIR}")
        shutil.rmtree(MODEL_CACHE_DIR)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    subprocess.run("pip install -q -U huggingface_hub", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")
        target = f"{MODEL_CACHE_DIR}/{model['dir']}"
        os.makedirs(target, exist_ok=True)

        ok = 0
        for filename in model["files"]:
            rel_path = filename
            # 从 filename 提取 bare name 用于保存
            bare_name = os.path.basename(filename)
            src_repo = model["id"]

            if download_file(src_repo, filename, target):
                ok += 1

        size = get_dir_size_gb(target)
        log(f"  结果: {model['dir']} ({size:.2f}GB, {ok}/{len(model['files'])}个)")

    # 最终
    log(f"\n{'='*55}")
    log("下载完成！")
    total = get_dir_size_gb(MODEL_CACHE_DIR)
    log(f"模型总计: {total:.2f}GB")
    for model in MODELS:
        path = f"{MODEL_CACHE_DIR}/{model['dir']}"
        size = get_dir_size_gb(path)
        done = "✅" if size > 0.1 else "❌"
        log(f"  {done} {model['name']}: {size:.2f}GB")


if __name__ == "__main__":
    main()
