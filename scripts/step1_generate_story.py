#!/usr/bin/env python3
"""Step 1: 剧本生成 - Gemini API"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import json
import re


def generate_script(episode_num=1, genre="urban_romance", prev_summary=""):
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash", generation_config={
        "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192,
    })

    prompt = f"""你是一个专业的中文短剧编剧。请为一部{genre}题材的AI短剧写第{episode_num}集的完整剧本。

角色: 小明(28岁程序员,内向善良,戴眼镜短发) | 小丽(26岁设计师,活泼开朗,长发) | 王总(45岁总监,严厉公正)
场景: office(现代办公室) cafe(温馨咖啡馆) park(城市公园) apartment(温馨公寓) street(城市街道)

要求: 3-5分钟, 5-8场景, 每场2-4镜头, 完整故事线+悬念结尾

纯JSON输出:
{{"episode": {episode_num}, "title": "标题", "scenes": [{{"scene_id": "scene_1", "location": "office",
"time_of_day": "morning", "lighting": "自然光", "mood": "氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 3, "description": "画面描述", "character": "xiaoming",
"action": "动作", "dialogue": "对话", "narration": "旁白",
"emotion": "情绪", "subtitle": "字幕"}}]}}], "next_episode_hook": "下集预告"}}"""

    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return json.loads(m.group())
        raise


def main():
    parser = argparse.ArgumentParser(description="AI短剧剧本生成")
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--genre", default="urban_romance")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("[ERROR] 设置 GOOGLE_API_KEY 环境变量")
        return

    print(f"生成第{args.episode}集剧本...")
    script = generate_script(args.episode, args.genre)

    os.makedirs(args.output_dir, exist_ok=True)
    path = f"{args.output_dir}/episode_{args.episode:02d}_script.json"
    save_json(script, path)
    print(f"[OK] {path} | {script.get('title')}")
    return script


if __name__ == "__main__":
    main()
