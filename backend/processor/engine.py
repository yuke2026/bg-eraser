"""图片处理引擎 — 去背景/加水印/改尺寸/批量"""

import io
import zipfile
from typing import Any
from PIL import Image, ImageFilter

from rembg import remove, new_session

import numpy as np
import cv2

# 全局 rembg session
_session = new_session("u2net")

MAX_FILE_SIZE = 10 * 1024 * 1024


def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, 255)


def _enhance_alpha_edge(alpha: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """锐化 alpha 通道的边缘过渡 — 用 Unsharp Mask 减少过渡区宽度。

    rembg 输出的 alpha 通道在边缘处通常是渐变过渡（软边缘），
    原图有水印/文字等元素时过渡区更宽更模糊。

    本函数对 alpha 通道做 Unsharp Mask（非 RGB），精准增强边缘
    梯度，缩小过渡区。不碰 RGB 像素，只影响透明度过渡。

    Args:
        alpha: (H, W) uint8 — 0=透明, 255=不透明
        amount: 0.0=不变, 0.5~1.5=建议范围

    Returns:
        增强后的 alpha 通道
    """
    if amount <= 0:
        return alpha

    blurred = cv2.GaussianBlur(alpha, (0, 0), 2)  # sigma=2
    alpha_f = alpha.astype(np.float32)
    sharpened = alpha_f + amount * (alpha_f - blurred.astype(np.float32))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def remove_background(image_bytes: bytes) -> bytes:
    """去背景 → 返回透明PNG bytes（带边缘增强后处理）"""
    # 1. rembg 推理 → RGBA
    result_rgba = remove(image_bytes, session=_session)

    # 2. 提取 alpha 通道做边缘增强
    img = Image.open(io.BytesIO(result_rgba)).convert("RGBA")
    r, g, b, a = img.split()

    alpha_np = np.array(a, dtype=np.uint8)

    # 判断是否需要增强：检测 alpha 过渡区像素比例
    edge_pixels = np.sum((alpha_np > 30) & (alpha_np < 225))
    total = alpha_np.shape[0] * alpha_np.shape[1]
    edge_ratio = edge_pixels / total

    if edge_ratio > 0.01:
        # 过渡区明显 → 用 Unsharp Mask 锐化边缘
        alpha_enhanced = _enhance_alpha_edge(alpha_np, amount=1.0)
        # 形态学清理：闭运算填小孔
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha_enhanced = cv2.morphologyEx(alpha_enhanced, cv2.MORPH_CLOSE, kernel)
    else:
        alpha_enhanced = alpha_np

    # 3. 重组 RGBA
    enhanced_a = Image.fromarray(alpha_enhanced, mode="L")
    output = Image.merge("RGBA", (r, g, b, enhanced_a))

    out = io.BytesIO()
    output.save(out, format="PNG")
    return out.getvalue()


def change_bg_color(image_bytes: bytes, color: str = "#ffffff") -> bytes:
    """去背景 + 换背景色

    如果输入已是透明 RGBA（已去过背景），直接合成背景色，
    避免重复跑 rembg 导致质量下降。
    """
    # 检测是否已是透明图
    try:
        test_img = Image.open(io.BytesIO(image_bytes))
        is_already_transparent = test_img.mode == "RGBA"
    except Exception:
        is_already_transparent = False

    if is_already_transparent:
        img = test_img.convert("RGBA")
    else:
        # 使用增强版去背景
        nobg = remove_background(image_bytes)
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
    opacity: float = 1.0,
    size_ratio: float = 0.75,
    margin: int | None = None,
) -> bytes:
    """加 logo 水印 — 在最终尺寸上渲染，清晰不模糊"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")

    # 缩放 logo — 基于最终图片尺寸
    logo_w = max(int(img.width * size_ratio), 20)
    logo_h = int(logo.height * (logo_w / logo.width))
    logo = logo.resize((logo_w, logo_h), Image.LANCZOS)

    # 边距 = 图片宽度的 2%（最小 8px）
    m = margin if margin is not None else max(int(img.width * 0.02), 8)

    # 位置
    positions = {
        "top-left": (m, m),
        "top-right": (img.width - logo_w - m, m),
        "bottom-left": (m, img.height - logo_h - m),
        "bottom-right": (img.width - logo_w - m, img.height - logo_h - m),
        "center": ((img.width - logo_w) // 2, (img.height - logo_h) // 2),
    }
    pos = positions.get(position, positions["top-left"])

    # 透明度
    r, g, b, a = logo.split()
    a = a.point(lambda x: int(x * opacity))
    logo = Image.merge("RGBA", (r, g, b, a))

    # 叠加
    img.paste(logo, pos, logo)
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def rotate_image(image_bytes: bytes, degrees: int = 90) -> bytes:
    """旋转图片 — 支持 90/180/270 度顺时针旋转，或自动转正(360=EXIF修正)。

    当 degrees=360 时，读取 EXIF Orientation 标签，自动转正。
    其他值：正数=顺时针, 负数=逆时针
    """
    img = Image.open(io.BytesIO(image_bytes))

    if degrees == 360:
        # 自动转正：读取 EXIF Orientation
        try:
            exif = img._getexif()
            if exif:
                orientation = exif.get(0x0112, 1)
                # EXIF: 1=正常, 3=180°, 6=顺时针90°, 8=逆时针90°
                # PIL rotate() 正数=逆时针(CCW), 负数=顺时针(CW)
                if orientation == 3:
                    degrees = 180
                elif orientation == 6:
                    degrees = -90   # 顺时针90°
                elif orientation == 8:
                    degrees = 90    # 逆时针90°
                else:
                    degrees = 0
        except Exception:
            degrees = 0
        if degrees == 0:
            return image_bytes  # 无需旋转

    # expand=True 确保旋转后不裁剪
    rotated = img.rotate(-degrees, expand=True, resample=Image.BICUBIC)
    out = io.BytesIO()
    rotated.save(out, format="PNG")
    return out.getvalue()


def process_pipeline(
    image_bytes: bytes,
    actions: list[dict[str, Any]],
    logo_bytes: bytes | None = None,
) -> bytes:
    """按操作链依次处理一张图片。
    自动确保 resize 在 add_watermark 之前执行，保证水印在最终尺寸上渲染、清晰不模糊。
    """
    # 重排：确保 resize 在 add_watermark 之前
    watermark_action = None
    clean_actions = []
    for step in actions:
        if step["action"] == "add_watermark" and logo_bytes:
            watermark_action = step
        else:
            clean_actions.append(step)
    if watermark_action:
        clean_actions.append(watermark_action)

    data = image_bytes
    for step in clean_actions:
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
        elif action == "rotate":
            deg = int(params.get("degrees", 90))
            data = rotate_image(data, deg)
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
