"""商品渲染引擎 — 使用 SiliconFlow API (硅基流动) 生成产品渲染图

支持：
- 文生图：根据 prompt 生成渲染结果
- 合成渲染：AI 生成场景背景 + 原商品图合成（商品像素级无损）
"""

import os
import io
import json
import time
import hashlib
import base64
import logging
import requests
import numpy as np
from PIL import Image
from typing import Optional

logger = logging.getLogger("render-engine")

# 配置
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_API_BASE = "https://api.siliconflow.cn/v1"

# 默认模型
DEFAULT_MODEL = "Kwai-Kolors/Kolors"  # 快速、高质量
ALT_MODEL = "Qwen/Qwen-Image"  # 备选模型


# ── 任务存储 ──

class RenderTaskStore:
    """内存中的渲染任务存储（生产环境应改用 Redis/数据库）"""

    def __init__(self):
        self._tasks = {}

    def create(self, task_id: str, prompt: str, size: tuple[int, int],
               ref_image: Optional[bytes] = None) -> dict:
        task = {
            "id": task_id,
            "status": "pending",
            "prompt": prompt,
            "size": size,
            "ref_image": ref_image,
            "result_bytes": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs):
        task = self._tasks.get(task_id)
        if task:
            task.update(kwargs)
            task["updated_at"] = time.time()

    def get_all(self) -> list[dict]:
        return list(self._tasks.values())


task_store = RenderTaskStore()


# ── SiliconFlow API 工具函数 ──


def _siliconflow_headers() -> dict:
    """获取 SiliconFlow API 请求头"""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError(
            "SILICONFLOW_API_KEY 未设置。\n"
            "请设置环境变量：export SILICONFLOW_API_KEY=你的Key"
        )
    return {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }


def _generate_image(prompt: str, size: tuple[int, int],
                    ref_image_bytes: Optional[bytes] = None,
                    model: str = DEFAULT_MODEL) -> bytes:
    """调用 SiliconFlow 文生图 API，返回图片 bytes"""
    width, height = size

    body = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
    }

    # 不传 image 参数 — 纯文生图，AI 不接触商品像素
    # 商品保留由合成渲染层 (_composite_with_mask) 保证

    resp = requests.post(
        f"{SILICONFLOW_API_BASE}/images/generations",
        headers=_siliconflow_headers(),
        json=body,
        timeout=120,
    )

    if not resp.ok:
        error_detail = resp.text[:500]
        logger.error(f"SiliconFlow API 错误: {resp.status_code} {error_detail}")
        raise RuntimeError(f"生成图片失败 ({resp.status_code}): {error_detail}")

    data = resp.json()

    images = data.get("images", [])
    if not images:
        raise RuntimeError("API 返回为空，没有生成图片")

    image_url = images[0].get("url")
    if not image_url:
        raise RuntimeError("API 返回格式异常")

    # 下载图片
    img_resp = requests.get(image_url, timeout=60)
    img_resp.raise_for_status()
    return img_resp.content


# ── 渲染逻辑（像素级保真） ──

def _generate_scene_background(prompt: str, size: tuple[int, int],
                                model: str = DEFAULT_MODEL) -> bytes:
    """AI 纯文生图生成场景背景（不传参考图，不改变商品）"""
    return _generate_image(prompt=prompt, size=size, ref_image_bytes=None, model=model)


def _extract_product_mask(image: Image.Image) -> np.ndarray:
    """从商品照片中提取前景（商品）mask

    适用于常见商品摄影（浅色/纯色背景）：
    1. LAB 色彩空间：背景通常是浅色 (L>90)，商品有丰富颜色
    2. 自适应阈值 + 形态学去噪
    """
    import cv2
    arr = np.array(image.convert("RGB"))

    # 转为 LAB 色彩空间 — L 通道区分亮度
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0].astype(np.float32)

    # Otsu 自动找分割阈值（亮度）
    L_uint8 = L.astype(np.uint8)
    _, mask_bg = cv2.threshold(L_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    product_mask = 255 - mask_bg  # 反转：商品 = 暗区

    # 饱和度线索：商品有色彩（高饱和度），背景低饱和度
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]
    sat = np.sqrt(a_channel.astype(np.float32)**2 + b_channel.astype(np.float32)**2)
    _, sat_bg = cv2.threshold(sat.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 合并亮度+饱和度
    combined = cv2.bitwise_and(product_mask, sat_bg)
    if combined.sum() < 1000:
        combined = product_mask  # 回退到亮度

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    # 取最大连通区域（商品主体）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    if num_labels > 1:
        areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)]
        if areas:
            label = max(areas, key=lambda x: x[0])[1]
            combined = np.where(labels == label, 255, 0).astype(np.uint8)

    # 膨胀让包裹更完整
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.dilate(combined, dilate_kernel, iterations=1)

    # 边缘羽化
    combined = cv2.GaussianBlur(combined, (15, 15), 3)

    return combined


def _composite_with_mask(original: Image.Image, background: Image.Image,
                         mask: np.ndarray) -> Image.Image:
    """用 mask 合成：商品区域用原图，背景区域用AI场景"""
    orig_rgba = original.convert("RGBA")
    bg_rgba = background.convert("RGBA")

    mask_float = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 4, axis=2)

    orig_arr = np.array(orig_rgba, dtype=np.float32)
    bg_arr = np.array(bg_rgba, dtype=np.float32)

    result_arr = (orig_arr * mask_3ch + bg_arr * (1.0 - mask_3ch)).astype(np.uint8)
    return Image.fromarray(result_arr, "RGBA")


def analyze_and_render(image_bytes: Optional[bytes], prompt: str,
                       width: int = 1024, height: int = 1024) -> dict:
    """渲染：AI 生成场景背景 + 原商品图合成（像素级保真）

    有参考图时：
      1. AI 纯文生图生成场景背景（不接触商品）
      2. LAB+Luminance 提取商品 mask
      3. 原商品图叠加到场景背景上
    无参考图时：纯文生图
    """
    analysis = {}

    if image_bytes:
        logger.info("合成渲染: 生成场景背景 + 提取商品 mask...")
    else:
        logger.info("无参考图，文生图模式")
        qualifiers = ["专业产品摄影", "高清细节", "柔和自然光"]
        if not any(q in prompt for q in ["高清", "细节", "专业", "摄影"]):
            prompt = prompt + "，专业产品摄影，高清细节，柔和自然光"

    logger.info(f"渲染prompt: {prompt[:120]}")

    task_id = hashlib.md5(f"{time.time()}{prompt}".encode()).hexdigest()[:12]
    task = task_store.create(
        task_id=task_id,
        prompt=prompt,
        size=(width, height),
        ref_image=image_bytes,
    )

    try:
        if image_bytes:
            from PIL import Image as PILImage

            orig_img = PILImage.open(io.BytesIO(image_bytes))

            # Step 1: AI 生成场景背景（纯文生图）
            logger.info("AI 生成场景背景中...")
            bg_bytes = _generate_scene_background(prompt, (width, height))
            bg_img = PILImage.open(io.BytesIO(bg_bytes)).convert("RGBA")
            bg_img = bg_img.resize((width, height), PILImage.LANCZOS)

            # Step 2: 提取商品 mask
            logger.info("提取商品 mask...")
            orig_resized = orig_img.resize((width, height), PILImage.LANCZOS)
            product_mask = _extract_product_mask(orig_resized)

            # Step 3: 合成
            logger.info("合成中...")
            result_img = _composite_with_mask(orig_resized, bg_img, product_mask)

            out = io.BytesIO()
            result_img.save(out, format="PNG")
            result_bytes = out.getvalue()
            logger.info(f"合成渲染完成: {width}x{height}")
        else:
            result_bytes = _generate_image(
                prompt=prompt,
                size=(width, height),
                ref_image_bytes=None,
                model=DEFAULT_MODEL,
            )

        task_store.update(task_id,
                          status="completed",
                          scene_prompt=prompt,
                          result_bytes=result_bytes,
                          analysis=analysis)

    except Exception as e:
        logger.exception("渲染失败")
        if image_bytes:
            try:
                logger.warning(f"主方案失败，尝试备选模型: {e}")
                from PIL import Image as PILImage
                bg_bytes = _generate_scene_background(prompt, (width, height), model=ALT_MODEL)
                bg_img = PILImage.open(io.BytesIO(bg_bytes)).convert("RGBA")
                bg_img = bg_img.resize((width, height), PILImage.LANCZOS)
                orig_resized = PILImage.open(io.BytesIO(image_bytes)).resize((width, height), PILImage.LANCZOS)
                product_mask = _extract_product_mask(orig_resized)
                result_img = _composite_with_mask(orig_resized, bg_img, product_mask)
                out = io.BytesIO()
                result_img.save(out, format="PNG")
                result_bytes = out.getvalue()
                task_store.update(task_id, status="completed",
                                  scene_prompt=prompt, result_bytes=result_bytes, analysis=analysis)
            except Exception as e2:
                task_store.update(task_id, status="failed",
                                  error=f"主方案失败: {e}; 备选也失败: {e2}")
        else:
            task_store.update(task_id, status="failed", error=str(e))

    return task_store.get(task_id)


def poll_render_status(task_id: str) -> dict:
    """获取渲染任务状态（SiliconFlow 同步API，任务直接完成或失败）"""
    task = task_store.get(task_id)
    if not task:
        return {"status": "not_found"}

    now = time.time()
    elapsed = (task.get("updated_at") or now) - (task.get("created_at") or now)
    result = {
        "task_id": task_id,
        "status": task["status"],
        "scene_prompt": task.get("scene_prompt", ""),
        "progress": 100 if task["status"] == "completed" else 0,
        "error": task.get("error"),
        "elapsed": max(elapsed, 0),
    }

    if task.get("result_bytes"):
        result["result_bytes"] = task["result_bytes"]

    return result
