#!/usr/bin/env python3
"""
Wan2.2 TI2V 5B AI短剧端到端流水线
===================================
在Kaggle Notebook中运行此脚本，自动完成：
  1. 剧本生成 (Gemini API / 本地 Qwen)
  2. 分镜生成 (结构化JSON)
  3. 视频生成 (Wan2.2 TI2V 5B via ComfyUI，跳过 SD 1.5)
  4. 配音生成 (ChatTTS / edge-tts)
  5. 剪辑合成 (FFmpeg)

与 kaggle-ai-series 的区别：
  - 跳过 Step 3 (SD 1.5 画面生成)
  - Step 4 使用 Wan2.2 TI2V 5B 直接文本生成视频
  - 模型路径: /kaggle/input/saysnkaggle/wan2-2-5b-f16/

Kaggle 运行:
  !rm -rf /kaggle/working/* && cd /kaggle/working && git clone https://github.com/szchengmi/video_wan2_2_5B_ti2v.git && cd video_wan2_2_5B_ti2v/scripts && python kaggle_pipeline.py --force
"""

import sys, os
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

# 导入 common 中的配置和工具
from common import (
    BASE_DIR, EPISODE_NUM, WAN22_DATASET, WAN22_MODELS_DIR,
    get_dirs, setup_dirs, log, run_cmd, save_json, load_json,
)

import time
import shutil
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V AI短剧生成")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--episode", type=int, default=None)
    args = parser.parse_args()

    if args.episode is not None:
        global EPISODE_NUM
        EPISODE_NUM = args.episode

    log("╔══════════════════════════════════════════╗")
    log("║   AI短剧 — Wan2.2 TI2V 5B Pipeline       ║")
    log("╚══════════════════════════════════════════╝")
    log(f"集数: {EPISODE_NUM} | 模型: {WAN22_DATASET}")

    # 清除旧输出
    if args.force:
        log("⚠️ 强制重新生成")
        ep_dir = get_dirs(EPISODE_NUM)["episode"]
        if os.path.isdir(ep_dir):
            shutil.rmtree(ep_dir)
            log(f"  已清除: {ep_dir}")

    setup_dirs()

    # 安装依赖
    log("\n安装依赖...")
    subprocess.run("pip install -q edge-tts psutil", shell=True, timeout=60)

    t0 = time.time()

    # Step 1: 剧本生成
    log("\n" + "=" * 50)
    log("Step 1: 剧本生成")
    log("=" * 50)
    from step1_generate_story import generate_script
    script = generate_script(EPISODE_NUM)
    log(f"剧本: {script.get('title')}")

    # Step 2: 分镜生成
    log("\n" + "=" * 50)
    log("Step 2: 分镜生成")
    log("=" * 50)
    from step2_generate_storyboard import generate_storyboard
    storyboard = generate_storyboard(script)
    total = sum(len(s.get("shots", [])) for s in storyboard.get("scenes", []))
    log(f"分镜: {len(storyboard['scenes'])}场景 | {total}镜头")

    # Step 3: 跳过 (Wan2.2 直接生成视频，不需要先画图)
    log("\n" + "=" * 50)
    log("Step 3: 画面生成 → 跳过 (Wan2.2 直接 T2V)")
    log("=" * 50)

    # Step 4: 视频生成 (Wan2.2 TI2V)
    log("\n" + "=" * 50)
    log("Step 4: 视频生成 (Wan2.2 TI2V 5B)")
    log("=" * 50)
    sb_path = f"{get_dirs(EPISODE_NUM)['storyboard']}/episode_{EPISODE_NUM:02d}_storyboard.json"
    videos_dir = f"{get_dirs(EPISODE_NUM)['videos']}"
    subprocess.run([
        "python", "step4_generate_videos_wan22.py",
        "--storyboard", sb_path,
        "--output-dir", videos_dir,
    ], cwd=_SCRIPT_DIR)

    # Step 5: 配音生成
    log("\n" + "=" * 50)
    log("Step 5: 配音生成")
    log("=" * 50)
    audio_dir = f"{get_dirs(EPISODE_NUM)['audio']}"
    subprocess.run([
        "python", "step5_generate_audio.py",
        "--storyboard", sb_path,
        "--output-dir", audio_dir,
    ], cwd=_SCRIPT_DIR)

    # Step 6: 剪辑合成
    log("\n" + "=" * 50)
    log("Step 6: 剪辑合成")
    log("=" * 50)
    final_dir = f"{get_dirs(EPISODE_NUM)['final']}"
    subprocess.run([
        "python", "step6_compose.py",
        "--storyboard", sb_path,
        "--videos-dir", videos_dir,
        "--audio-dir", audio_dir,
        "--output-dir", final_dir,
    ], cwd=_SCRIPT_DIR)

    # 总结
    elapsed = (time.time() - t0) / 60
    log(f"\n{'=' * 50}")
    log(f"全部完成! 耗时: {elapsed:.1f} 分钟")
    log(f"输出: {final_dir}")
    log(f"{'=' * 50}")


if __name__ == "__main__":
    main()
