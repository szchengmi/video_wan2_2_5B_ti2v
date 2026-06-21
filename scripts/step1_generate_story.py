#!/usr/bin/env python3
"""Step 1: 剧本生成 - Gemini API"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

import argparse
import json
import re


def generate_script(episode_num=1, genre="urban_romance", prev_summary="",
                    duration_minutes=3, style="二次元"):
    from google import genai

    client = genai.Client(api_key=GOOGLE_API_KEY)

    # Gemini 根据总时长灵活决定场景数和镜头数
    # 每个镜头 duration_seconds 在剧本中指定，总时长 ≈ 所有镜头之和
    duration_sec = int(duration_minutes)  # 参数名保留但实际是秒

    prompt = f"""你是一个专业的中文短剧编剧和声音设计师。请为一部{genre}题材的AI短剧写第{episode_num}集的完整剧本。

角色: 小明(28岁程序员,内向善良,戴眼镜短发) | 小丽(26岁设计师,活泼开朗,长发) | 王总(45岁总监,严厉公正)
场景: office(现代办公室) cafe(温馨咖啡馆) park(城市公园) apartment(温馨公寓) street(城市街道)

═══════════════════════════════════════════
节奏规划原则（重要！）
═══════════════════════════════════════════
1. 镜头时长由内容决定，不是固定值：
   - 对话密集场景：4-8 秒（让观众看清画面+听清对话）
   - 纯环境/过渡镜头：2-4 秒
   - 动作/紧张场景：3-5 秒，可快速切换
   - 建立镜头（每场第一个）：稍长 4-6 秒
2. 场景规划：
   - 不要频繁换场景，一个场景内可以有多个长镜头
   - 用不同景别讲述故事：建立→对话→反应→特写，避免无意义切镜
   - 总镜头数由 Gemini 根据故事节奏自然决定，不要刻意凑数也不要刻意限制
   - 长镜头优先：一个镜头可以持续 6-12 秒，给观众沉浸感
   - 只有当视角/地点/时间确实需要变化时才切换镜头
3. 留白与呼吸：
   - 不要填满整个 duration，留 10-15% 给沉默/留白
   - 对话之间有 0.3-0.8 秒的停顿间隔
   - 镜头切换处留 0.2-0.5 秒黑场过渡

═══════════════════════════════════════════
音频设计原则（重要！）
═══════════════════════════════════════════
每个镜头必须设计声音层次，用 audio_events 对象详细描述：

audio_events 包含 4 个子数组（必须全部存在，无内容写空数组 []）：

1. dialogue（角色对话）：有角色台词时填写
   - role: 角色拼音 (xiaoming/xiaoli/boss_wang)
   - lines: 台词内容
   - timecode: "开始秒-结束秒" 格式，如 "0.5-2.3"
   - emotion: 情绪（参考下方情绪词）
   - speed: 语速偏移，正数=快，负数=慢，如 +3/-1/0

2. VO（旁白）：有旁白时填写
   - role: 固定 "narrator"
   - lines: 旁白内容
   - timecode: 时间范围
   - emotion: 情绪
   - speed: 语速偏移

3. SFX（音效）：有音效时填写（如雷声、关门、脚步声、手机响等）
   - sound: 音效名称标识
   - path: 音效文件路径，如 "sound/door_close.mp3"
   - timecode: 时间范围
   - volume: 音量 0.0-1.0

4. Atmos（环境音）：有环境音时填写（如街道噪音、雨声、室内空调声等）
   - sound: 环境音名称标识
   - path: 音效文件路径，如 "sound/rain.mp3"
   - timecode: 时间范围（通常覆盖整个镜头）
   - volume: 音量 0.0-1.0
   - loop: 是否循环播放 true/false

时间码规则：
- 从 0.0 开始计算，第一个声音元素从 0.0 或稍后开始
- 对话之间留 0.3-0.8 秒间隔
- 所有时间码不能超过 duration_seconds
- 最后一个声音结束后留 0.3-0.5 秒静音

情绪词库：happy/sad/angry/surprised/nervous/calm/determined/embarrassed/thoughtful/紧张/感激/抒情/平静/恐惧/愤怒/温柔/急迫

duration_seconds 计算：
- Gemini 根据音频时间码自行计算：duration = max(所有事件结束时间) + 0.5s 留白
- 确保 duration 合理（2-8秒之间），与镜头内容匹配

═══════════════════════════════════════════
纯JSON输出（不要 markdown 代码块）
═══════════════════════════════════════════
{{"episode": {episode_num}, "title": "标题", "style": "{style}", "scenes": [{{"scene_id": "scene_1", "location": "office",
"time_of_day": "morning", "lighting": "自然光", "mood": "氛围",
"shots": [{{"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
"duration_seconds": 5.0,
"description": "画面描述", "character": "xiaoming",
"action": "动作", "emotion": "calm",
"audio_events": {{"dialogue": [{{"role": "xiaoming", "lines": "台词", "timecode": "0.5-2.3", "emotion": "紧张", "speed": +3}}],
"VO": [], "SFX": [], "Atmos": [{{"sound": "office_ambience", "path": "sound/office.mp3", "timecode": "0.0-5.0", "volume": 0.2, "loop": true}}]}},
"subtitle": "字幕文本"}}], "next_episode_hook": "下集预告"}}]}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
        config={
            "temperature": 0.9, "top_p": 0.95, "top_k": 40, "max_output_tokens": 16384,
        },
    )
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 修复截断JSON：补全未闭合的花括号
        fixed = text.rstrip().rstrip(',')
        opens = fixed.count('{')
        closes = fixed.count('}')
        if opens > closes:
            fixed += '}' * (opens - closes)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                return json.loads(m.group())
            # 最终降级：返回预置剧本
            return _get_fallback_script(episode_num, genre)


def _get_fallback_script(episode_num=1, genre="urban_romance", style="二次元"):
    """预置兜底剧本（Gemini JSON 解析失败时使用）—— 兼容 audio_events 结构"""
    return {
        "episode": episode_num,
        "title": "第一集：初遇",
        "style": style,
        "scenes": [
            {
                "scene_id": "scene_1", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "平静日常",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 4.5, "description": "小明在办公室敲代码", "character": "xiaoming",
                     "action": "专注地敲击键盘", "emotion": "calm",
                     "dialogue": "", "narration": "周一的早晨，办公室里只有键盘的声音。",
                     "subtitle": "周一的早晨，办公室里只有键盘的声音。",
                     "audio_events": {
                         "dialogue": [],
                         "VO": [{"role": "narrator", "lines": "周一的早晨，办公室里只有键盘的声音。",
                                  "timecode": "0.3-4.0", "emotion": "平静", "speed": -1}],
                         "SFX": [],
                         "Atmos": [{"sound": "office_ambience", "path": "sound/office.mp3",
                                     "timecode": "0.0-4.5", "volume": 0.15, "loop": True}]
                     }},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 3.0, "description": "小明表情特写", "character": "xiaoming",
                     "action": "微微皱眉", "emotion": "thoughtful",
                     "dialogue": "这个需求又改了...", "narration": "",
                     "subtitle": "这个需求又改了...",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "这个需求又改了...",
                                        "timecode": "0.5-2.5", "emotion": "thoughtful", "speed": 0}],
                         "VO": [], "SFX": [], "Atmos": []
                     }},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 5.0, "description": "小丽走进办公室", "character": "xiaoli",
                     "action": "推门进来，笑着打招呼", "emotion": "happy",
                     "dialogue": "早啊小明！今天天气真好！", "narration": "",
                     "subtitle": "早啊小明！今天天气真好！",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoli", "lines": "早啊小明！今天天气真好！",
                                        "timecode": "0.5-2.8", "emotion": "happy", "speed": +2}],
                         "VO": [],
                         "SFX": [{"sound": "door_open", "path": "sound/door_open.mp3",
                                   "timecode": "0.0-0.3", "volume": 0.5}],
                         "Atmos": []
                     }}
                ]
            },
            {
                "scene_id": "scene_2", "location": "cafe", "time_of_day": "afternoon",
                "lighting": "暖黄灯光", "mood": "温馨浪漫",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 6.0, "description": "小明和小丽在咖啡馆聊天", "character": "xiaoming",
                     "action": "端着咖啡杯，认真倾听", "emotion": "calm",
                     "dialogue": "你觉得这个设计方案怎么样？", "narration": "",
                     "subtitle": "你觉得这个设计方案怎么样？",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "你觉得这个设计方案怎么样？",
                                        "timecode": "0.5-3.5", "emotion": "calm", "speed": 0}],
                         "VO": [],
                         "SFX": [{"sound": "coffee_cup", "path": "sound/coffee_cup.mp3",
                                   "timecode": "0.0-0.2", "volume": 0.3}],
                         "Atmos": [{"sound": "cafe_ambience", "path": "sound/cafe.mp3",
                                     "timecode": "0.0-6.0", "volume": 0.2, "loop": True}]
                     }},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 4.0, "description": "小丽眼睛发亮", "character": "xiaoli",
                     "action": "眼睛发亮，兴奋地比划", "emotion": "happy",
                     "dialogue": "我觉得配色可以再大胆一些！", "narration": "",
                     "subtitle": "我觉得配色可以再大胆一些！",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoli", "lines": "我觉得配色可以再大胆一些！",
                                        "timecode": "0.5-3.0", "emotion": "happy", "speed": +1}],
                         "VO": [], "SFX": [], "Atmos": []
                     }},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "pan_right",
                     "duration_seconds": 5.0, "description": "两人相视而笑", "character": "xiaoming",
                     "action": "忍不住笑了", "emotion": "embarrassed",
                     "dialogue": "你总是这么有想法。", "narration": "",
                     "subtitle": "你总是这么有想法。",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "你总是这么有想法。",
                                        "timecode": "1.0-3.5", "emotion": "embarrassed", "speed": 0}],
                         "VO": [{"role": "narrator", "lines": "咖啡的温暖也比不上彼此眼中的光芒。",
                                  "timecode": "3.5-4.8", "emotion": "温柔", "speed": -1}],
                         "SFX": [], "Atmos": []
                     }}
                ]
            },
            {
                "scene_id": "scene_3", "location": "office", "time_of_day": "evening",
                "lighting": "夕阳余晖", "mood": "紧张",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 6.0, "description": "王总走进办公室", "character": "boss_wang",
                     "action": "严肃地推门进来", "emotion": "angry",
                     "dialogue": "小明，客户对方案很不满意！", "narration": "",
                     "subtitle": "小明，客户对方案很不满意！",
                     "audio_events": {
                         "dialogue": [{"role": "boss_wang", "lines": "小明，客户对方案很不满意！",
                                        "timecode": "0.5-3.5", "emotion": "愤怒", "speed": +3}],
                         "VO": [],
                         "SFX": [{"sound": "door_slam", "path": "sound/door_slam.mp3",
                                   "timecode": "0.0-0.3", "volume": 0.6}],
                         "Atmos": []
                     }},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 4.0, "description": "小明紧张的表情", "character": "xiaoming",
                     "action": "紧张地站起来", "emotion": "nervous",
                     "dialogue": "什么？我明明按需求做的...", "narration": "",
                     "subtitle": "什么？我明明按需求做的...",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "什么？我明明按需求做的...",
                                        "timecode": "0.5-3.0", "emotion": "恐惧", "speed": +4}],
                         "VO": [], "SFX": [], "Atmos": []
                     }},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "static",
                     "duration_seconds": 5.0, "description": "三人对峙", "character": "boss_wang",
                     "action": "将文件摔在桌上", "emotion": "angry",
                     "dialogue": "需求已经变了，你不知道吗？明天早上之前改好！", "narration": "",
                     "subtitle": "需求已经变了，你不知道吗？明天早上之前改好！",
                     "audio_events": {
                         "dialogue": [{"role": "boss_wang", "lines": "需求已经变了，你不知道吗？明天早上之前改好！",
                                        "timecode": "0.5-4.0", "emotion": "愤怒", "speed": +2}],
                         "VO": [],
                         "SFX": [{"sound": "paper_slap", "path": "sound/paper_slap.mp3",
                                   "timecode": "0.3-0.5", "volume": 0.5}],
                         "Atmos": []
                     }}
                ]
            },
            {
                "scene_id": "scene_4", "location": "apartment", "time_of_day": "night",
                "lighting": "台灯光", "mood": "温馨感人",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 6.0, "description": "小明在公寓加班", "character": "xiaoming",
                     "action": "疲惫地盯着屏幕", "emotion": "sad",
                     "dialogue": "", "narration": "夜深了，小明还在改方案。",
                     "subtitle": "夜深了，小明还在改方案。",
                     "audio_events": {
                         "dialogue": [],
                         "VO": [{"role": "narrator", "lines": "夜深了，小明还在改方案。",
                                  "timecode": "0.5-5.0", "emotion": "sad", "speed": -2}],
                         "SFX": [],
                         "Atmos": [{"sound": "night_quiet", "path": "sound/night.mp3",
                                     "timecode": "0.0-6.0", "volume": 0.15, "loop": True}]
                     }},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 5.0, "description": "小丽端着夜宵进来", "character": "xiaoli",
                     "action": "轻轻推门，端着夜宵", "emotion": "calm",
                     "dialogue": "还没休息？我给你带了宵夜。", "narration": "",
                     "subtitle": "还没休息？我给你带了宵夜。",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoli", "lines": "还没休息？我给你带了宵夜。",
                                        "timecode": "0.5-3.5", "emotion": "温柔", "speed": -1}],
                         "VO": [],
                         "SFX": [{"sound": "door_creak", "path": "sound/door_creak.mp3",
                                   "timecode": "0.0-0.2", "volume": 0.3}],
                         "Atmos": []
                     }},
                    {"shot_id": "shot_3", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 4.0, "description": "小明感动地看着小丽", "character": "xiaoming",
                     "action": "感动地看着小丽", "emotion": "happy",
                     "dialogue": "谢谢你，小丽。有你在真好。", "narration": "",
                     "subtitle": "谢谢你，小丽。有你在真好。",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "谢谢你，小丽。有你在真好。",
                                        "timecode": "0.5-3.0", "emotion": "感激", "speed": 0}],
                         "VO": [], "SFX": [], "Atmos": []
                     }}
                ]
            },
            {
                "scene_id": "scene_5", "location": "park", "time_of_day": "morning",
                "lighting": "阳光明媚", "mood": "充满希望",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "wide_shot", "camera_movement": "pan_left",
                     "duration_seconds": 6.0, "description": "公园里晨跑", "character": "xiaoming",
                     "action": "在公园晨跑", "emotion": "calm",
                     "dialogue": "", "narration": "改完方案的第二天，小明决定出门透透气。",
                     "subtitle": "改完方案的第二天，小明决定出门透透气。",
                     "audio_events": {
                         "dialogue": [],
                         "VO": [{"role": "narrator", "lines": "改完方案的第二天，小明决定出门透透气。",
                                  "timecode": "0.5-5.0", "emotion": "平静", "speed": -1}],
                         "SFX": [],
                         "Atmos": [{"sound": "park_morning", "path": "sound/park.mp3",
                                     "timecode": "0.0-6.0", "volume": 0.25, "loop": True}]
                     }},
                    {"shot_id": "shot_2", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 5.0, "description": "偶遇小丽", "character": "xiaoli",
                     "action": "惊喜地挥手", "emotion": "surprised",
                     "dialogue": "小明！好巧啊！", "narration": "",
                     "subtitle": "小明！好巧啊！",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoli", "lines": "小明！好巧啊！",
                                        "timecode": "0.5-2.5", "emotion": "surprised", "speed": +5}],
                         "VO": [], "SFX": [], "Atmos": []
                     }},
                    {"shot_id": "shot_3", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 6.0, "description": "两人并肩走在公园", "character": "xiaoming",
                     "action": "并肩散步，相视而笑", "emotion": "happy",
                     "dialogue": "小丽，昨晚的方案客户通过了！", "narration": "",
                     "subtitle": "小丽，昨晚的方案客户通过了！",
                     "audio_events": {
                         "dialogue": [{"role": "xiaoming", "lines": "小丽，昨晚的方案客户通过了！",
                                        "timecode": "0.5-3.5", "emotion": "happy", "speed": +1}],
                         "VO": [{"role": "narrator", "lines": "阳光正好，好像一切都在变好。",
                                  "timecode": "4.0-5.5", "emotion": "温柔", "speed": -1}],
                         "SFX": [], "Atmos": []
                     }}
                ]
            },
            {
                "scene_id": "scene_6", "location": "office", "time_of_day": "morning",
                "lighting": "自然光", "mood": "紧张期待",
                "shots": [
                    {"shot_id": "shot_1", "shot_type": "medium_shot", "camera_movement": "static",
                     "duration_seconds": 5.0, "description": "王总宣布消息", "character": "boss_wang",
                     "action": "站在会议室前方", "emotion": "calm",
                     "dialogue": "告诉大家一个好消息——", "narration": "",
                     "subtitle": "告诉大家一个好消息——",
                     "audio_events": {
                         "dialogue": [{"role": "boss_wang", "lines": "告诉大家一个好消息——",
                                        "timecode": "0.5-3.5", "emotion": "calm", "speed": 0}],
                         "VO": [], "SFX": [],
                         "Atmos": [{"sound": "office_ambience", "path": "sound/office.mp3",
                                     "timecode": "0.0-5.0", "volume": 0.15, "loop": True}]
                     }},
                    {"shot_id": "shot_2", "shot_type": "close_up", "camera_movement": "static",
                     "duration_seconds": 3.5, "description": "小明和小丽紧张对视", "character": "xiaoming",
                     "action": "紧张地握紧拳头", "emotion": "nervous",
                     "dialogue": "", "narration": "",
                     "subtitle": "",
                     "audio_events": {
                         "dialogue": [],
                         "VO": [],
                         "SFX": [],
                         "Atmos": [{"sound": "tension", "path": "sound/tension.mp3",
                                     "timecode": "0.0-3.5", "volume": 0.2, "loop": True}]
                     }},
                    {"shot_id": "shot_3", "shot_type": "wide_shot", "camera_movement": "dolly_in",
                     "duration_seconds": 5.0, "description": "王总微笑", "character": "boss_wang",
                     "action": "露出罕见的微笑", "emotion": "happy",
                     "dialogue": "客户非常满意！小明、小丽，你们做到了！", "narration": "",
                     "subtitle": "客户非常满意！小明、小丽，你们做到了！",
                     "audio_events": {
                         "dialogue": [{"role": "boss_wang", "lines": "客户非常满意！小明、小丽，你们做到了！",
                                        "timecode": "0.5-4.0", "emotion": "happy", "speed": +1}],
                         "VO": [],
                         "SFX": [{"sound": "applause", "path": "sound/applause.mp3",
                                   "timecode": "3.5-4.5", "volume": 0.4}],
                         "Atmos": []
                     }}
                ]
            }
        ],
        "next_episode_hook": "小明和小丽的项目获得了成功，但新的挑战正在等着他们..."
    }


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
