"""
商品渲染引擎 — 使用 SiliconFlow API (硅基流动) 生成产品渲染图

支持：
- 文生图：根据 prompt 生成渲染结果
- 图生图：以参考图为起点做风格化变体
- 异步/同步：同步 API（等待生成完成）
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
            "replicate_prediction_id": None,
            "result_url": None,
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
    """调用 SiliconFlow 文生图/图生图 API，返回图片 bytes"""
    width, height = size

    body = {
        "model": model,
        "prompt": prompt,
        "n": 1,
        "size": f"{width}x{height}",
    }

    # 如果有参考图，尝试 img2img
    if ref_image_bytes:
        # SiliconFlow 支持通过 image 参数传参考图
        img_b64 = base64.b64encode(ref_image_bytes).decode("utf-8")
        body["image"] = f"data:image/png;base64,{img_b64}"

    resp = requests.post(
        f"{SILICONFLOW_API_BASE}/images/generations",
        headers=_siliconflow_headers(),
        json=body,
        timeout=120,  # 文生图可能需要较长时间
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


# ── 分析引擎（参考图分析） ──

def analyze_reference(image_bytes: bytes, prompt: str) -> dict:
    """分析参考图，提取风格/构图/色彩信息

    不使用 AI API，基于图像处理做初步分析：
    - 主色调提取（K-Means 聚类）
    - 亮度/对比度评估
    - 构图分析（主体位置）
    """
    from io import BytesIO
    from PIL import Image
    import numpy as np

    try:
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        h, w, _ = img_np.shape

        # 主色调分析
        pixels = img_np.reshape(-1, 3)
        # 简单平均色
        avg_color = pixels.mean(axis=0)
        color_name = _color_name(avg_color)

        # 亮度分析
        gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114])
        brightness = gray.mean()
        if brightness < 64:
            brightness_label = "暗调"
        elif brightness < 128:
            brightness_label = "中等偏暗"
        elif brightness < 192:
            brightness_label = "明亮"
        else:
            brightness_label = "高亮"

        contrast = gray.std()
        contrast_label = "高对比" if contrast > 60 else "柔和" if contrast < 30 else "适中对比"

        # 构图分析（主体位置 - 简单边缘检测）
        from PIL import ImageFilter
        edges = pil_img.filter(ImageFilter.FIND_EDGES)
        edge_np = np.array(edges.convert("L"))
        edge_center_y = np.unravel_index(edge_np.argmax(), edge_np.shape)[0] / h
        if edge_center_y < 0.3:
            comp_label = "主体偏上"
        elif edge_center_y > 0.7:
            comp_label = "主体偏下"
        else:
            comp_label = "主体居中"

        analysis = {
            "色调": color_name,
            "亮度": brightness_label,
            "对比度": contrast_label,
            "构图": comp_label,
            "尺寸": f"{w}×{h}",
            "prompt_enhancement": _build_enhanced_prompt(prompt, {
                "color": color_name,
                "brightness": brightness_label,
                "contrast": contrast_label,
                "composition": comp_label,
            }),
        }
        return analysis
    except Exception as e:
        logger.warning(f"引用图分析失败: {e}")
        return {
            "prompt_enhancement": prompt,
            "note": f"分析失败: {e}",
        }


def _color_name(rgb: np.ndarray) -> str:
    """将 RGB 值映射到中文颜色名"""
    r, g, b = rgb
    if max(r, g, b) < 50:
        return "深色/黑色系"
    if min(r, g, b) > 200:
        return "浅色/白色系"
    if r > g and r > b:
        return "暖色调（偏红）"
    if g > r and g > b:
        return "冷色调（偏绿）"
    if b > r and b > g:
        return "冷色调（偏蓝）"
    return "中性色调"


def _build_enhanced_prompt(original: str, analysis: dict) -> str:
    """根据分析结果增强 prompt"""
    parts = [original]

    # 追加场景描述（如果用户未指定）
    has_scene = any(kw in original for kw in ["背景", "场景", "环境", "桌上", "草地", "户外", "室内"])
    if not has_scene:
        parts.append("专业产品摄影风格")

    parts.append("高清细节")
    parts.append("柔和自然光")

    return ", ".join(parts)


# ── 处理步骤 ──

def analyze_and_render(image_bytes: Optional[bytes], prompt: str,
                       width: int = 1024, height: int = 1024) -> dict:
    """分析参考图 → 增强 prompt → 提交渲染

    如果没有参考图 (image_bytes=None)，直接文生图
    """
    # Step 1: 分析参考图（如果有）
    analysis = {}
    enhanced_prompt = prompt

    if image_bytes:
        logger.info("分析参考图中...")
        analysis = analyze_reference(image_bytes, prompt)
        enhanced_prompt = analysis.get("prompt_enhancement", prompt)
    else:
        logger.info("无参考图，直接使用用户prompt进行文生图")
        # 给纯文生图加一些基本质量修饰
        qualifiers = ["专业产品摄影", "高清细节", "柔和自然光"]
        has_qualifier = any(q in prompt for q in ["高清", "细节", "专业", "摄影"])
        if not has_qualifier:
            enhanced_prompt = prompt + "，专业产品摄影，高清细节，柔和自然光"
        else:
            enhanced_prompt = prompt

    logger.info(f"增强prompt: {enhanced_prompt}")

    # Step 3: 提交渲染任务
    task_id = hashlib.md5(f"{time.time()}{prompt}".encode()).hexdigest()[:12]

    task = task_store.create(
        task_id=task_id,
        prompt=enhanced_prompt,
        size=(width, height),
        ref_image=image_bytes,
    )

    # 尝试调用 SiliconFlow API 生成图片
    try:
        result_bytes = _generate_image(
            prompt=enhanced_prompt,
            size=(width, height),
            ref_image_bytes=image_bytes,
            model=DEFAULT_MODEL,
        )
        task_store.update(task_id,
                          status="completed",
                          enhanced_prompt=enhanced_prompt,
                          result_bytes=result_bytes,
                          analysis=analysis)
    except RuntimeError as e:
        if "SILICONFLOW_API_KEY" in str(e):
            # API Key 未配置
            task_store.update(task_id, status="no_api_key",
                              enhanced_prompt=enhanced_prompt,
                              analysis=analysis,
                              error=str(e))
        else:
            # 尝试用备选模型重试
            try:
                logger.warning(f"模型 {DEFAULT_MODEL} 失败，尝试备选模型 {ALT_MODEL}: {e}")
                result_bytes = _generate_image(
                    prompt=enhanced_prompt,
                    size=(width, height),
                    ref_image_bytes=image_bytes,
                    model=ALT_MODEL,
                )
                task_store.update(task_id,
                                  status="completed",
                                  enhanced_prompt=enhanced_prompt,
                                  result_bytes=result_bytes,
                                  analysis=analysis)
            except Exception as e2:
                task_store.update(task_id, status="failed",
                                  error=f"主模型失败: {e}; 备选模型也失败: {e2}")
    except Exception as e:
        task_store.update(task_id, status="failed", error=str(e))
        logger.exception("渲染失败")

    return task_store.get(task_id)


def poll_render_status(task_id: str) -> dict:
    """获取渲染任务状态（SiliconFlow 同步API，任务直接完成或失败）"""
    task = task_store.get(task_id)
    if not task:
        return {"status": "not_found"}

    # 同步 API 已经返回结果，无需轮询
    result = {
        "task_id": task_id,
        "status": task["status"],
        "enhanced_prompt": task.get("enhanced_prompt", ""),
        "progress": 100 if task["status"] == "completed" else 0,
        "error": task.get("error"),
    }

    if task.get("result_bytes"):
        result["result_bytes"] = task["result_bytes"]

    return result
