"""指令解析器 — 自然语言 → 操作链"""

import re
from typing import Any

# 意图词表
INTENT_PATTERNS: list[tuple[str, str]] = [
    (r"去背景|抠图|去底|移除背景|抠出|去掉背景", "remove_bg"),
    (r"加水印|加logo|加图标|加商标|加品牌", "add_watermark"),
    (r"(\d+)\s*[×xX*]\s*(\d+)", "resize"),
    (r"换(.{1,4})背景|改成(.{1,4})底|背景变|背景改成", "change_bg_color"),
    (r"拼图|合并|合成|拼接|拼在一起", "composite"),
    (r"批量|所有图|全部图|每张", "batch_flag"),
    (r"旋转|转正|翻转|顺时针|逆时针|竖屏|横屏|倒过来", "rotate"),
]

POSITION_PATTERNS: list[tuple[str, str]] = [
    (r"左上|左上角", "top-left"),
    (r"右上|右上角", "top-right"),
    (r"左下|左下角", "bottom-left"),
    (r"右下|右下角|右下角", "bottom-right"),
    (r"居中|中间|中心", "center"),
]

COLOR_MAP: dict[str, str] = {
    "白": "#ffffff",
    "黑": "#000000",
    "红": "#ff0000",
    "蓝": "#0000ff",
    "绿": "#00ff00",
    "黄": "#ffff00",
    "灰": "#808080",
    "透明": "transparent",
}


def parse_prompt(prompt: str) -> dict[str, Any]:
    """
    解析用户自然语言指令，返回操作链。

    >>> parse_prompt("去背景 + 加logo在右下角 + 800×800")
    {
        "actions": [
            {"action": "remove_bg", "params": {}},
            {"action": "add_watermark", "params": {"position": "bottom-right"}},
            {"action": "resize", "params": {"width": 800, "height": 800}}
        ],
        "is_batch": False
    }
    """
    text = prompt.strip()
    if not text:
        return {"actions": [], "is_batch": False}

    actions: list[dict[str, Any]] = []
    is_batch = False
    seen_actions: set[str] = set()

    for pattern, intent in INTENT_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue

        if intent == "batch_flag":
            is_batch = True
            continue
        if intent in seen_actions:
            continue
        seen_actions.add(intent)

        action: dict[str, Any] = {"action": intent, "params": {}}

        # 位置参数（适用于加水印）
        if intent == "add_watermark":
            for p_pat, pos in POSITION_PATTERNS:
                if re.search(p_pat, text):
                    action["params"]["position"] = pos
                    break
            if "position" not in action["params"]:
                action["params"]["position"] = "top-left"

        # 旋转参数
        if intent == "rotate":
            # 检测角度数值
            deg_match = re.search(r"(\d+)\s*度", text)
            if deg_match:
                deg = int(deg_match.group(1))
            elif re.search(r"顺|右", text):
                deg = 90
            elif re.search(r"逆|左", text):
                deg = -90
            elif re.search(r"180|倒过|翻转", text):
                deg = 180
            else:
                deg = 90  # 默认顺时针90度
            action["params"]["degrees"] = deg

        # 尺寸参数
        if intent == "resize":
            m = re.search(r"(\d+)\s*[×xX*]\s*(\d+)", text)
            if m:
                action["params"]["width"] = int(m.group(1))
                action["params"]["height"] = int(m.group(2))

        # 颜色参数
        if intent == "change_bg_color":
            for cn, code in COLOR_MAP.items():
                if cn in text:
                    action["params"]["color"] = code
                    break
            if "color" not in action["params"]:
                action["params"]["color"] = "#ffffff"

        actions.append(action)

    return {"actions": actions, "is_batch": is_batch}


# 快捷模板（顺序：去背景→改尺寸→加水印，确保水印在最终尺寸上渲染）
TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "taobao": [
        {"action": "remove_bg", "params": {}},
        {"action": "resize", "params": {"width": 800, "height": 800, "mode": "contain"}},
        {"action": "add_watermark", "params": {"position": "top-left"}},
    ],
    "jd": [
        {"action": "remove_bg", "params": {}},
        {"action": "resize", "params": {"width": 1200, "height": 1200, "mode": "contain"}},
        {"action": "add_watermark", "params": {"position": "top-left"}},
    ],
    "pdd": [
        {"action": "remove_bg", "params": {}},
        {"action": "change_bg_color", "params": {"color": "#ffffff"}},
        {"action": "resize", "params": {"width": 750, "height": 750, "mode": "contain"}},
        {"action": "add_watermark", "params": {"position": "top-left"}},
    ],
    "douyin": [
        {"action": "remove_bg", "params": {}},
        {"action": "resize", "params": {"width": 1080, "height": 1920, "mode": "contain"}},
        {"action": "add_watermark", "params": {"position": "top-left"}},
    ],
}


def get_template(name: str) -> list[dict[str, Any]] | None:
    """获取快捷模板"""
    return TEMPLATES.get(name)


def get_templates_list() -> dict:
    """获取所有模板列表"""
    return {
        "taobao": {"name": "淘宝主图", "desc": "去背景+水印+800×800", "icon": "🛍"},
        "jd": {"name": "京东商品图", "desc": "去背景+水印+1200×1200", "icon": "🏪"},
        "pdd": {"name": "拼多多白底图", "desc": "去背景+白底+水印+750×750", "icon": "🎯"},
        "douyin": {"name": "抖音商品卡", "desc": "去背景+水印+1080×1920", "icon": "📱"},
    }
