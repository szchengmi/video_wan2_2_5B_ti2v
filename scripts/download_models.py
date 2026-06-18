#!/usr/bin/env python3
"""
Kaggle 模型下载脚本 v13
====================
直接指定每个模型要下载的文件名，不依赖list_repo_tree过滤
逐个下载，实时容量监控
"""

import os
import sys
import time
import shutil
import subprocess

MODEL_CACHE_DIR = "/kaggle/working/kaggle-ai-series/models"
# 如果Dataset已挂载，优先下载到Dataset路径
if os.path.isdir("/kaggle/input/newdataset/kaggle-ai-series"):
    MODEL_CACHE_DIR = "/kaggle/input/newdataset/kaggle-ai-series/models"

def get_kaggle_secret(key_name):
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass
    try:
        if key_name == "HF_TOKEN":
            return secret_value_1  # noqa: F821
    except:
        pass
    return os.environ.get(key_name, "")

HF_TOKEN = get_kaggle_secret("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_HUB_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    print(f"[OK] HF_TOKEN: {HF_TOKEN[:10]}...")
else:
    print("[WARN] HF_TOKEN 未设置")

os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def get_disk_free_gb():
    import shutil as _s
    _, _, free = _s.disk_usage("/kaggle/working")
    return free / 1e9

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

def check_capacity(label=""):
    free = get_disk_free_gb()
    used = get_dir_size_gb(MODEL_CACHE_DIR)
    log(f"📊 {label}剩余: {free:.1f}GB | models: {used:.2f}GB")
    return free

def is_model_ready(dir_name, min_size_mb=100):
    target = f"{MODEL_CACHE_DIR}/{dir_name}"
    if not os.path.exists(target):
        return False
    for f in os.listdir(target):
        fp = f"{target}/{f}"
        if f.endswith(('.safetensors', '.bin', '.gguf')) and os.path.isfile(fp):
            if os.path.getsize(fp) > min_size_mb * 1024 * 1024:
                return True
    return False

def download_file(model_id, filename, dest_path):
    """下载单个文件，直接到目标位置"""
    from huggingface_hub import hf_hub_download
    try:
        dest_dir = os.path.dirname(dest_path)
        os.makedirs(dest_dir, exist_ok=True)
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=dest_dir,
        )
        return True
    except Exception as e:
        log(f"    ❌ {filename}: {e}")
        return False

def download_model(model_id, dir_name, files):
    """按指定文件列表下载模型"""
    target = f"{MODEL_CACHE_DIR}/{dir_name}"

    if is_model_ready(dir_name):
        size = get_dir_size_gb(target)
        log(f"  ✅ {dir_name} 已存在 ({size:.2f}GB)")
        return True

    log(f"  ⬇️  {model_id} ({len(files)}个文件)")
    os.makedirs(target, exist_ok=True)
    t0 = time.time()

    ok = 0
    for filename in files:
        dest = f"{target}/{filename}"
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            ok += 1
            log(f"    ✅ {filename} (已存在)")
            continue

        if download_file(model_id, filename, dest):
            ok += 1
            size_mb = os.path.getsize(dest) / 1e6
            log(f"    ✅ {filename} ({size_mb:.0f}MB)")
        else:
            log(f"    ❌ {filename}")

        # 每个文件后检查容量
        free = get_disk_free_gb()
        if free < 0.5:
            log(f"  ⚠️  磁盘不足！剩余{free:.1f}GB，停止下载")
            break

    elapsed = time.time() - t0
    size = get_dir_size_gb(target)

    if ok > 0 and size > 0.1:
        log(f"  ✅ {dir_name} ({size:.2f}GB, {ok}/{len(files)}个, {elapsed:.0f}秒)")
        return True
    else:
        log(f"  ❌ 失败")
        return False


# ============================================================
# 模型定义 — 直接指定文件名
# ============================================================

MODELS = [
    {
        "id": "runwayml/stable-diffusion-v1-5",
        "name": "SD 1.5",
        "dir": "stable-diffusion-v1-5",
        "desc": "~4.27GB",
        "files": [
            "v1-5-pruned-emaonly.safetensors",
            "model_index.json",
        ],
    },
    {
        "id": "guoyww/animatediff-motion-adapter-v1-5-2",
        "name": "AnimateDiff",
        "dir": "animatediff",
        "desc": "~301MB",
        "files": [
            "diffusion_pytorch_model.safetensors",
            "config.json",
        ],
    },
    {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "name": "Qwen2.5-3B",
        "dir": "Qwen2.5-3B-Instruct",
        "desc": "~6.44GB",
        "files": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "merges.txt",
        ],
    },
]


# ============================================================
# 主流程
# ============================================================

def main():
    log("=" * 55)
    log("  Kaggle AI短剧 - 模型下载 v13 (指定文件名)")
    log("=" * 55)
    log(f"目标: {MODEL_CACHE_DIR}")
    check_capacity("初始 ")

    subprocess.run("pip install -q -U huggingface_hub", shell=True, timeout=120)

    for i, model in enumerate(MODELS, 1):
        log(f"\n{'='*55}")
        log(f"[{i}/{len(MODELS)}] {model['name']} ({model['desc']})")
        download_model(model["id"], model["dir"], model["files"])
        check_capacity(f"#{i} ")

    # 最终结果
    log(f"\n{'='*55}")
    log("全部完成！")
    total = get_dir_size_gb(MODEL_CACHE_DIR)
    free = get_disk_free_gb()
    log(f"模型总计: {total:.2f}GB | 磁盘剩余: {free:.1f}GB")

    for model in MODELS:
        path = f"{MODEL_CACHE_DIR}/{model['dir']}"
        size = get_dir_size_gb(path)
        done = "✅" if size > 0.1 else "❌"
        log(f"  {done} {model['name']}: {size:.2f}GB")

    log(f"\n✅ Save as Dataset → kaggle-ai-series-models")
    log("=" * 55)


if __name__ == "__main__":
    main()
