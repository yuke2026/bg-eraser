"""图片处理引擎 — 去背景/加水印/改尺寸/批量"""

import io
import zipfile
from typing import Any
from PIL import Image

from rembg import remove, new_session

# 全局 rembg session
_session = new_session("u2net")

MAX_FILE_SIZE = 10 * 1024 * 1024


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, 255)


def remove_background(image_bytes: bytes) -> bytes:
    """去背景 → 返回透明PNG bytes"""
    return remove(image_bytes, session=_session)


def change_bg_color(image_bytes: bytes, color: str = "#ffffff") -> bytes:
    """去背景 + 换背景色"""
    nobg = remove(image_bytes, session=_session)
    img = Image.open(io.BytesIO(nobg)).convert("RGBA")
    bg = Image.new("RGBA", img.size, _hex_to_rgba(color))
    composite = Image.alpha_composite(bg, img).convert("RGB")
    out = io.BytesIO()
    composite.save(out, format="PNG")
    return out.getvalue()


def resize_image(
    image_bytes: bytes,
    width: int,
    height: int,
    mode: str = "cover",
    bg_color: str = "#ffffff",
) -> bytes:
    """调整尺寸
    - cover: 裁剪填充（先缩放至完全覆盖，再居中裁剪）
    - contain: 留白填充（缩放到完全容纳，留白填色）
    - stretch: 直接拉伸
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGBA") if img.mode != "RGBA" else img

    if mode == "cover":
        ratio = max(width / img.width, height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        left = (img.width - width) // 2
        top = (img.height - height) // 2
        img = img.crop((left, top, left + width, top + height))

    elif mode == "contain":
        canvas = Image.new("RGBA", (width, height), _hex_to_rgba(bg_color))
        r = min(width / img.width, height / img.height)
        resized = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)
        x = (width - resized.width) // 2
        y = (height - resized.height) // 2
        canvas.paste(resized, (x, y), resized if resized.mode == "RGBA" else None)
        img = canvas

    else:  # stretch
        img = img.resize((width, height), Image.LANCZOS)

    out = io.BytesIO()
    img.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def add_watermark(
    image_bytes: bytes,
    logo_bytes: bytes,
    position: str = "top-left",
    opacity: float = 0.8,
    size_ratio: float = 0.75,
    margin: int = 20,
) -> bytes:
    """加 logo 水印"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")

    # 缩放 logo
    logo_w = int(img.width * size_ratio)
    logo_h = int(logo.height * (logo_w / logo.width))
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # 位置
    positions = {
        "top-left": (margin, margin),
        "top-right": (img.width - logo_w - margin, margin),
        "bottom-left": (margin, img.height - logo_h - margin),
        "bottom-right": (img.width - logo_w - margin, img.height - logo_h - margin),
        "center": ((img.width - logo_w) // 2, (img.height - logo_h) // 2),
    }
    pos = positions.get(position, positions["bottom-right"])

    # 透明度
    r, g, b, a = logo.split()
    a = a.point(lambda x: int(x * opacity))
    logo = Image.merge("RGBA", (r, g, b, a))

    # 叠加
    img.paste(logo, pos, logo)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def process_pipeline(
    image_bytes: bytes,
    actions: list[dict[str, Any]],
    logo_bytes: bytes | None = None,
) -> bytes:
    """按操作链依次处理一张图片"""
    data = image_bytes
    for step in actions:
        action = step["action"]
        params = step.get("params", {})

        if action == "remove_bg":
            data = remove_background(data)
        elif action == "change_bg_color":
            data = change_bg_color(data, params.get("color", "#ffffff"))
        elif action == "resize":
            w = params.get("width", 800)
            h = params.get("height", 800)
            data = resize_image(data, w, h, params.get("mode", "contain"))
        elif action == "add_watermark":
            if logo_bytes:
                data = add_watermark(
                    data, logo_bytes,
                    position=params.get("position", "top-left"),
                    size_ratio=params.get("size_ratio", 0.75),
                )
        elif action == "composite":
            # 简单拼接 — 需要多张图支持
            pass
    return data


def process_batch(
    images: list[tuple[str, bytes]],
    actions: list[dict[str, Any]],
    logo_bytes: bytes | None = None,
) -> list[tuple[str, bytes, float]]:
    """批量处理多张图片，返回 [(filename, bytes, processing_time), ...]"""
    import time
    results = []
    for name, data in images:
        t0 = time.perf_counter()
        result = process_pipeline(data, actions, logo_bytes)
        elapsed = time.perf_counter() - t0
        base = name.rsplit(".", 1)[0] if "." in name else name
        results.append((f"{base}_processed.png", result, elapsed))
    return results


def create_zip(results: list[tuple[str, bytes, float]]) -> bytes:
    """将处理结果打包为 ZIP"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data, _ in results:
            zf.writestr(name, data)
    buf.seek(0)
    return buf.getvalue()
