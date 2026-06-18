#!/usr/bin/env python3
"""
Wan2.2 TI2V 5B 模型下载
====================
Kaggle 上直接下载模型到 /kaggle/working/models/

用法:
  !python download_models.py

依赖: 已配置 Kaggle Secret HF_TOKEN
"""
import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

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


def download_file(model_id, filename, dest_path):
    """下载单个文件，直接到目标位置"""
    from huggingface_hub import hf_hub_download

    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    # 已存在且足够大则跳过
    if os.path.isfile(dest_path) and os.path.getsize(dest_path) > 100 * 1024 * 1024:
        size_mb = os.path.getsize(dest_path) / 1e6
        log(f"  ✅ {os.path.basename(dest_path)} (已存在, {size_mb:.0f}MB)")
        return True

    log(f"  ⬇️ {os.path.basename(dest_path)}...")

    try:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=dest_dir,
        )
        if os.path.isfile(dest_path):
            size_mb = os.path.getsize(dest_path) / 1e6
            log(f"  ✅ 完成: {size_mb:.0f}MB")
            return True
        else:
            log(f"  ❌ 下载后文件不存在")
            return False
    except Exception as e:
        log(f"  ❌ {e}")
        return False


# ============================================================
# 模型定义 — 直接指定 repo_id 和 filename
# ============================================================

MODELS = [
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 TI2V 5B UNET",
        "dir": "wan22_ti2v_5b",
        "desc": "~10GB",
        "files": [
            ("split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_ti2v_5B_fp16.safetensors"),
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "name": "Wan2.2 VAE",
        "dir": "wan22_vae",
        "desc": "~1.5GB",
        "files": [
            ("split_files/vae/wan2.2_vae.safetensors", "wan2.2_vae.safetensors"),
        ],
    },
    {
        "id": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "name": "UMT5 XXL Encoder",
        "dir": "umt5_xxl",
        "desc": "~12GB",
        "files": [
            ("split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
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

    subprocess.run("pip install -q -U huggingface_hub", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")
        target = f"{MODEL_CACHE_DIR}/{model['dir']}"
        os.makedirs(target, exist_ok=True)

        ok = 0
        for filename, save_name in model["files"]:
            dest = f"{target}/{save_name}"

            if download_file(model["id"], filename, dest):
                ok += 1

            free = shutil.disk_usage("/kaggle/working").free / 1e9
            if free < 0.5:
                log(f"  ⚠️  磁盘不足！剩余{free:.1f}GB，停止下载")
                break

        size = get_dir_size_gb(target)
        log(f"  结果: {model['dir']} ({size:.2f}GB, {ok}/{len(model['files'])}个)")

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
