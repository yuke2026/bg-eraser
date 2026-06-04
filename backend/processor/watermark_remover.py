"""水印检测 + 智能移除引擎

策略：
1. 自动检测：通过 MSER 文本聚类 + 边缘分析定位水印区域
2. 半自动：用户指定矩形区域 (x1,y1,x2,y2)
3. 修复：OpenCV Telea inpainting 填充缺失区域

设计原则：宁可漏检也不误伤 —— 检测保守，只返回高置信度区域。
纯 CPU 可运行，无需 GPU。
"""

import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import logging
logger = logging.getLogger("watermark-remover")

# ── 辅助函数 ─────────────────────────────────


def _pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL RGBA/RGB → OpenCV BGR"""
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _cv2_to_pil_bytes(cv_img: np.ndarray) -> bytes:
    """OpenCV BGR → PNG bytes"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _expand_rect(x: int, y: int, w: int, h: int, img_w: int, img_h: int, factor: float = 0.05) -> tuple[int, int, int, int]:
    """将矩形向外扩展 factor 比例，增大修复区域避免边缘残留"""
    dx = int(w * factor)
    dy = int(h * factor)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)
    return x1, y1, x2, y2


def _cluster_rects(
    rects: list[tuple[int, int, int, int]],
    img_w: int, img_h: int,
    distance_thresh: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    """将邻近矩形聚类合并成更大的区域

    基于中心距离（归一化到图片尺寸）聚类，而非 IoU。
    因为一行文字的字符不重叠，IoU 检测不到，但中心距离很近。
    """
    if not rects:
        return []

    # 提取中心点
    centers = np.array([[x + w // 2, y + h // 2] for x, y, w, h in rects], dtype=np.float32)
    sizes = np.array([w * h for x, y, w, h in rects], dtype=np.float32)
    diag = np.sqrt(img_w ** 2 + img_h ** 2)

    used = set()
    clusters = []

    for i, (cx, cy) in enumerate(centers):
        if i in used:
            continue

        # 找到所有与 i 邻近的矩形
        dx = np.abs(centers[:, 0] - cx) / img_w
        dy = np.abs(centers[:, 1] - cy) / img_h
        nearby = np.where((dx < distance_thresh) & (dy < distance_thresh))[0]

        cluster_indices = [idx for idx in nearby if idx not in used]
        if not cluster_indices:
            continue

        # 聚类结果 = 包含所有矩形的最小外接矩形
        xs = [rects[idx][0] for idx in cluster_indices]
        ys = [rects[idx][1] for idx in cluster_indices]
        x2s = [rects[idx][0] + rects[idx][2] for idx in cluster_indices]
        y2s = [rects[idx][1] + rects[idx][3] for idx in cluster_indices]

        cx1, cy1 = min(xs), min(ys)
        cx2, cy2 = max(x2s), max(y2s)

        for idx in cluster_indices:
            used.add(idx)

        clusters.append((cx1, cy1, cx2 - cx1, cy2 - cy1))

    return clusters


# ── 检测策略 ─────────────────────────────────


def _detect_text_clusters(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """MSER 检测文字 → 聚类合并 → 返回高置信度水印区域

    核心思路：真实水印文字具有「多个小字符连成一行/一块」的特征。
    单个孤立的小区域（如噪点）会被过滤。
    """
    img_h, img_w = gray.shape

    # Step 1: MSER 检测（参数放宽以捕获透明文字）
    mser = cv2.MSER_create(
        delta=5, min_area=20, max_area=5000,
        max_variation=0.5, min_diversity=0.3,
    )
    regions, _ = mser.detectRegions(gray)

    # Step 2: 过滤单个字符区域
    char_rects = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        area = w * h
        aspect = w / max(h, 1)

        # 真实文字：面积适中、宽高比合理
        # 过滤掉过正方块（网格交叉点典型特征）
        if 30 < area < 8000 and 0.2 < aspect < 10:
            # 过滤近正方形区域（网格交叉点）
            squareness = min(aspect, 1/aspect)
            if squareness > 0.85:  # 仅排除几乎完美的正方形
                continue
            char_rects.append((x, y, w, h))

    if len(char_rects) < 2:
        # 至少要有 2 个字符才可能是水印
        return []

    # Step 3: 聚类 — 将邻近字符合并为水印组
    clusters = _cluster_rects(char_rects, img_w, img_h, distance_thresh=0.25)

    # Step 4: 过滤聚类结果
    filtered = []
    img_area = img_w * img_h
    for x, y, w, h in clusters:
        area = w * h

        # 水印面积通常不大不小：0.05% ~ 15% 图片面积
        if area < 0.0005 * img_area or area > 0.15 * img_area:
            continue

        # 水印通常在边缘区域（距离图片边界较近）
        margin_x = min(x, img_w - x - w) / img_w
        margin_y = min(y, img_h - y - h) / img_h

        # 水印很少在图片正中央
        if margin_x > 0.35 and margin_y > 0.35:
            continue

        # 计算聚类区域内 MSER 点的填充密度 — 水印区域应该有密集的 MSER 点
        # 滤除网格线等在角落形成的稀疏聚集
        cluster_mask = np.zeros((h, w), dtype=np.uint8)
        for cx, cy, cw, ch in char_rects:
            # 如果字符中心在 cluster 内，标记
            ccx = cx + cw // 2
            ccy = cy + ch // 2
            if x <= ccx <= x + w and y <= ccy <= y + h:
                local_x = max(0, ccx - x)
                local_y = max(0, ccy - y)
                cluster_mask[local_y:min(local_y+ch, h), local_x:min(local_x+cw, w)] = 255

        fill_ratio = np.count_nonzero(cluster_mask) / max(h * w, 1)
        if fill_ratio < 0.05:  # 填充率太低 → 稀疏噪点，不是水印
            continue

        filtered.append((x, y, w, h))

    return filtered


def _detect_semitransparent_regions(
    hsv: np.ndarray, gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """检测半透明叠加水印

    半透明水印的视觉特征：图像中局部出现「不该有的边缘/纹理」。
    用 HSV 做初筛 + 边缘验证。
    """
    img_h, img_w = gray.shape
    _, s, v = cv2.split(hsv)
    s, v = s.astype(np.float32), v.astype(np.float32)

    # 半透明水印：饱和度偏低 + 亮度中等偏亮（对背景有轻微改变）
    low_sat = s < 50
    mid_val = (v > 60) & (v < 230)
    candidate = (low_sat & mid_val).astype(np.uint8) * 255

    # 形态学闭运算连接邻近区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    # 轮廓提取
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    img_area = img_w * img_h
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # 面积阈值：0.05% ~ 10% 图片面积
        if area < 0.0005 * img_area or area > 0.1 * img_area:
            continue

        # 宽高比不能太极端
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 10:
            continue

        # 边缘密度验证 — 文字区域应该有适中的纹理
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        edges = cv2.Canny(roi, 50, 200)
        edge_density = np.count_nonzero(edges) / max(roi.size, 1)

        # 有文字的区域应该有边缘（之前错误地过滤掉了有边缘的区域）
        if edge_density < 0.01:  # 纯色区域边缘太低 → 不是水印
            continue

        rects.append((x, y, w, h))

    return rects


def _detect_transparent_text(
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """通过梯度幅值滤波检测透明叠加文字水印

    原理：透明文字的边缘强度介于「产品强边缘」和「纹理噪声」之间。
    通过 Sobel 梯度幅值做带通滤波，保留中等强度的边缘，
    然后形态学闭运算连接文字笔画，最后轮廓提取。

    优势：不受文字颜色和透明度影响，对产品图片中的叠加水印尤其有效。
    """
    img_h, img_w = gray.shape

    # Step 1: Sobel 梯度幅值
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Step 2: 带通滤波 — 保留中等强度边缘，过滤强边缘（产品边界）和弱噪声
    moderate = ((mag > 4) & (mag < 35)).astype(np.uint8) * 255

    # Step 3: 形态学闭运算连接文字笔画
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(moderate, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 4: 开运算去除孤立噪点
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Step 5: 轮廓提取
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    img_area = img_w * img_h
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # 面积过滤
        if area < 0.001 * img_area or area > 0.20 * img_area:
            continue

        # 宽高比过滤（文字通常是矩形条状）
        aspect = w / max(h, 1)
        if aspect < 0.5 or aspect > 12:
            continue

        rects.append((x, y, w, h))

    return rects


def _detect_corner_logos(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """检测角落 logo/水印（通常在四角位置）"""
    h, w = gray.shape
    corner_rects = []

    # 四个角 ROI（取 25% 区域）
    rois = [
        ("top-left", (0, 0, int(w * 0.25), int(h * 0.25))),
        ("top-right", (int(w * 0.75), 0, int(w * 0.25), int(h * 0.25))),
        ("bottom-left", (0, int(h * 0.75), int(w * 0.25), int(h * 0.25))),
        ("bottom-right", (int(w * 0.75), int(h * 0.75), int(w * 0.25), int(h * 0.25))),
    ]

    for name, (rx, ry, rw, rh) in rois:
        roi = gray[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            continue

        edges = cv2.Canny(roi, 30, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        # 边缘密度适中（有内容但不过度纹理）
        if edge_density < 0.01 or edge_density > 0.2:
            continue

        # 自适应二值化找连通区域
        binary = cv2.adaptiveThreshold(
            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            roi_area = rw * rh

            # 过滤细长线条
            aspect = cw / max(ch, 1)
            if aspect > 8 or aspect < 1/8:
                continue

            if 0.01 * roi_area < area < 0.5 * roi_area:
                corner_rects.append((rx + cx, ry + cy, cw, ch))

    return corner_rects


def _detect_colored_text(
    hsv: np.ndarray,
    gray: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """检测特定颜色文字水印（浅红/浅灰等电商常见颜色）

    JD.COM 类浅红色水印：H 0-10 或 170-180, S 30-100, V 180-255
    """
    img_h, img_w = gray.shape
    h, s, v = cv2.split(hsv)

    # 红色在 HSV 中分布在两个区间：0-10 和 170-180
    red_mask1 = cv2.inRange(hsv, np.array([0, 30, 150]), np.array([12, 120, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([165, 30, 150]), np.array([180, 120, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 浅灰色/半透明白色：低饱和度 + 高亮度
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 40, 250]))

    # 合并颜色遮罩
    color_mask = cv2.bitwise_or(red_mask, gray_mask)

    # 形态学操作连接邻近色块
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # 轮廓提取
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        # 面积适中
        img_area = img_w * img_h
        if area < 0.001 * img_area or area > 0.20 * img_area:
            continue
        # 宽高比合理
        aspect = cw / max(ch, 1)
        if aspect < 0.5 or aspect > 12:
            continue
        # 在边缘区域（水印通常在图片边缘）
        margin_x = min(x, img_w - x - cw) / img_w
        margin_y = min(y, img_h - y - ch) / img_h
        if margin_x > 0.3 and margin_y > 0.3:
            continue
        rects.append((x, y, cw, ch))

    return rects


# ── 主检测函数 ─────────────────────────────────


def detect_watermarks(image_bytes: bytes) -> dict:
    """自动检测图片中的水印区域

    Returns:
        {regions: [{x,y,w,h,method}, ...], mask_b64: str, count: int}
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = _pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    img_h, img_w = gray.shape

    all_regions: list[tuple[int, int, int, int, str, float]] = []

    # 策略1: 文字聚类检测（最高优先级）
    text_clusters = _detect_text_clusters(gray)
    for x, y, w, h in text_clusters:
        confidence = min(1.0, (w * h) / (img_w * img_h * 0.05))
        all_regions.append((x, y, w, h, "text", 0.5 + confidence * 0.5))
    logger.info(f"text clusters: {len(text_clusters)}")

    # 策略2: 彩色文字检测（JD.COM 类浅红色水印）
    colored_rects = _detect_colored_text(hsv, gray)
    for x, y, w, h in colored_rects:
        all_regions.append((x, y, w, h, "colored", 0.65))
    logger.info(f"colored text: {len(colored_rects)}")

    # 策略3: 高频残差检测透明文字（通用型，不依赖颜色/透明度）
    text_highfreq = _detect_transparent_text(gray)
    for x, y, w, h in text_highfreq:
        all_regions.append((x, y, w, h, "highfreq", 0.55))
    logger.info(f"transparent text (highfreq): {len(text_highfreq)}")

    # 策略4: 角落 logo 检测
    corner_rects = _detect_corner_logos(gray)
    for x, y, w, h in corner_rects:
        all_regions.append((x, y, w, h, "corner", 0.6))
    logger.info(f"corner logos: {len(corner_rects)}")

    # 策略5: 半透明层检测（最低优先级）
    semi_rects = _detect_semitransparent_regions(hsv, gray)
    for x, y, w, h in semi_rects:
        all_regions.append((x, y, w, h, "semitransparent", 0.4))
    logger.info(f"semi-transparent: {len(semi_rects)}")

    if not all_regions:
        return {"regions": [], "count": 0}

    # 按置信度降序排列，取 top 5
    all_regions.sort(key=lambda r: r[5], reverse=True)
    top_regions = all_regions[:5]

    # 聚类合并（防止不同策略检测到同一区域）
    rects_no_score = [(x, y, w, h) for x, y, w, h, _, _ in top_regions]
    merged = _cluster_rects(rects_no_score, img_w, img_h, distance_thresh=0.06)

    # 构建返回
    regions = []
    for x, y, w, h in merged:
        ex1, ey1, ex2, ey2 = _expand_rect(x, y, w, h, img_w, img_h)
        regions.append({
            "x": int(ex1), "y": int(ey1),
            "w": int(ex2 - ex1), "h": int(ey2 - ey1),
            "method": "auto",
        })

    return {"regions": regions, "count": len(regions)}


# ── 简单擦除（替代 inpainting） ─────────────────


def _sample_border_color(
    cv_img: np.ndarray,
    x: int, y: int, w: int, h: int,
    border_width: int = 10,
) -> np.ndarray:
    """采样水印区域周围 border_width 像素的平均色"""
    img_h, img_w = cv_img.shape[:2]

    top = max(0, y - border_width)
    bottom = min(img_h, y + h + border_width)
    left = max(0, x - border_width)
    right = min(img_w, x + w + border_width)

    patches = []
    # 上边带
    if top < y:
        patches.append(cv_img[top:y, left:right])
    # 下边带
    if bottom > y + h:
        patches.append(cv_img[y + h:bottom, left:right])
    # 左边带
    if left < x:
        patches.append(cv_img[y:y + h, left:x])
    # 右边带
    if right > x + w:
        patches.append(cv_img[y:y + h, x + w:right])

    if not patches:
        return np.array([128, 128, 128], dtype=np.uint8)

    all_pixels = np.vstack([p.reshape(-1, 3) for p in patches])
    return np.mean(all_pixels, axis=0).astype(np.uint8)


def _gradient_fill(
    cv_img: np.ndarray,
    x: int, y: int, w: int, h: int,
    border_width: int = 10,
) -> np.ndarray:
    """渐变填充水印区域

    采样四边 border_width 像素带，生成从上到下、从左到右的渐变色填充。
    比纯色填充更自然，不会有「贴了一块」的感觉。
    """
    img_h, img_w = cv_img.shape[:2]

    # 裁剪边界
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    rw = x2 - x1
    rh = y2 - y1

    if rw <= 1 or rh <= 1:
        return cv_img

    result = cv_img.copy()

    # 采样四边颜色（保持像素级精度）
    top_color = None
    if y1 > 0:
        top_color = cv_img[y1 - 1, x1:x2]  # (rw, 3) — 每列一个颜色

    bottom_color = None
    if y2 < img_h:
        bottom_color = cv_img[y2, x1:x2]   # (rw, 3)

    left_color = None
    if x1 > 0:
        left_color = cv_img[y1:y2, x1 - 1]  # (rh, 3) — 每行一个颜色

    right_color = None
    if x2 < img_w:
        right_color = cv_img[y1:y2, x2]      # (rh, 3)

    # 双线性渐变填充
    for dy in range(rh):
        v_ratio = dy / max(rh - 1, 1)
        for dx in range(rw):
            h_ratio = dx / max(rw - 1, 1)

            color_sum = np.zeros(3, dtype=np.float32)
            weight_sum = 0.0

            if top_color is not None:
                w_t = 1.0 - v_ratio
                color_sum += top_color[dx].astype(np.float32) * w_t
                weight_sum += w_t

            if bottom_color is not None:
                w_b = v_ratio
                color_sum += bottom_color[dx].astype(np.float32) * w_b
                weight_sum += w_b

            if left_color is not None:
                w_l = 1.0 - h_ratio
                color_sum += left_color[dy].astype(np.float32) * w_l
                weight_sum += w_l

            if right_color is not None:
                w_r = h_ratio
                color_sum += right_color[dy].astype(np.float32) * w_r
                weight_sum += w_r

            if weight_sum > 0:
                result[y1 + dy, x1 + dx] = (color_sum / weight_sum).astype(np.uint8)

    return result


def _erase_watermark(
    image_bytes: bytes,
    x: int, y: int, w: int, h: int,
) -> bytes:
    """擦除水印：用周围背景的渐变填充矩形区域

    不做纹理合成，而是采样四边像素生成自然渐变，
    效果：水印文字消失，留下与周围过渡自然的底色。
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = _pil_to_cv2(pil_img)
    img_h, img_w = cv_img.shape[:2]

    # 扩展边界（留一点余量确保完全覆盖）
    ex1, ey1, ex2, ey2 = _expand_rect(x, y, w, h, img_w, img_h, factor=0.03)
    logger.info(f"erase region: ({ex1},{ey1})-({ex2},{ey2})")

    result = _gradient_fill(cv_img, ex1, ey1, ex2 - ex1, ey2 - ey1)

    return _cv2_to_pil_bytes(result)


def _erase_watermarks_batch(
    image_bytes: bytes,
    regions: list[dict],
) -> bytes:
    """批量擦除多个水印区域

    依次擦除每个区域（后擦的会参考先擦的结果）
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = _pil_to_cv2(pil_img)
    img_h, img_w = cv_img.shape[:2]

    result = cv_img.copy()
    for r in regions:
        rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
        ex1, ey1, ex2, ey2 = _expand_rect(rx, ry, rw, rh, img_w, img_h, factor=0.03)
        result = _gradient_fill(result, ex1, ey1, ex2 - ex1, ey2 - ey1)

    return _cv2_to_pil_bytes(result)


def remove_watermark_by_rect(
    image_bytes: bytes,
    x: int, y: int, w: int, h: int,
) -> bytes:
    """根据指定矩形区域擦除水印（替代原 inpainting 版本）"""
    return _erase_watermark(image_bytes, x, y, w, h)


def remove_watermarks_auto(
    image_bytes: bytes,
    exclude_regions: list[dict] | None = None,
) -> dict:
    """自动检测并擦除所有水印（可排除特定区域）"""
    detection = detect_watermarks(image_bytes)
    regions = detection["regions"]

    if not regions:
        return {"success": False, "result_bytes": None, "regions": [], "count": 0}

    # 排除用户标记忽略的区域
    if exclude_regions:
        keep = []
        for r in regions:
            excluded = False
            rx, ry, rw2, rh2 = r["x"], r["y"], r["w"], r["h"]
            for ex in exclude_regions:
                ex_x, ex_y = ex.get("x", 0), ex.get("y", 0)
                ex_w, ex_h = ex.get("w", 0), ex.get("h", 0)
                xx1 = max(rx, ex_x)
                yy1 = max(ry, ex_y)
                xx2 = min(rx + rw2, ex_x + ex_w)
                yy2 = min(rh2 + ry, ex_y + ex_h)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter > 0.2 * (rw2 * rh2):
                    excluded = True
                    break
            if not excluded:
                keep.append(r)
        regions = keep

    if not regions:
        return {"success": False, "result_bytes": None, "regions": [], "count": 0}

    result_bytes = _erase_watermarks_batch(image_bytes, regions)

    return {
        "success": True,
        "result_bytes": result_bytes,
        "regions": regions,
        "count": len(regions),
    }


# ── 文字水印搜索（用户输入文字定位） ─────────────────


def find_watermark_by_text(
    image_bytes: bytes,
    text: str,
) -> dict:
    """根据用户输入的文字内容，在图片中搜索水印位置

    使用多尺度模板匹配：把输入文字渲染成多个尺寸的图片，
    在目标图中搜索最佳匹配位置。

    Returns:
        {found: bool, x,y,w,h: 匹配位置, confidence: float}
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cv_img = _pil_to_cv2(pil_img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    if not text or not text.strip():
        return {"found": False}

    text = text.strip()

    # 在多个字体尺寸下搜索
    best_val = -1
    best_box = None

    for font_size in range(18, max(img_w, img_h) // 4, 4):
        # 渲染文字
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                font_size,
            )
        except Exception:
            font = ImageFont.load_default()

        # 创建文字的 PIL 图像（白色文字，黑色背景）
        dummy = Image.new("L", (1, 1), 0)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if tw > img_w - 30 or th > img_h - 30:
            break

        # 渲染文字模板（仅保留2px边距，框精确贴合文字）
        tmpl = Image.new("L", (tw + 4, th + 4), 0)
        tdraw = ImageDraw.Draw(tmpl)
        tdraw.text((2 - bbox[0], 2 - bbox[1]), text, fill=255, font=font)
        tmpl_np = np.array(tmpl, dtype=np.uint8)

        # 模板匹配
        result = cv2.matchTemplate(gray, tmpl_np, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_val:
            best_val = max_val
            best_box = (max_loc[0], max_loc[1], tw, th, max_val)

        # 如果匹配度已经很高，提前结束
        if max_val > 0.5:
            break

    if best_val < 0.15 or best_box is None:
        return {"found": False, "confidence": 0.0}

    x, y, tw, th, conf = best_box
    return {
        "found": True,
        "x": int(x),
        "y": int(y),
        "w": int(tw + 4),
        "h": int(th + 4),
        "confidence": round(float(conf), 3),
    }
