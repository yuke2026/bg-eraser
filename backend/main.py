"""BgEraser — V1 经典版 + V2 专业版"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove, new_session
from PIL import Image
import io, os, logging, time, uuid, json
from typing import Optional

from backend.nlu.parser import parse_prompt, get_templates_list, get_template
from backend.processor.engine import (
    process_batch, create_zip,
    remove_background, change_bg_color as engine_change_bg,
)
from backend.processor.watermark_remover import (
    detect_watermarks, remove_watermark_by_rect, remove_watermarks_auto,
    find_watermark_by_text, _erase_watermarks_batch,
)
from backend.render.engine import (
    analyze_and_render, poll_render_status, task_store,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bg-eraser")

def _fmt_time(seconds: float) -> str:
    """格式化秒数为可读字符串，如 '3.2s'"""
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"


app = FastAPI(title="BgEraser API", version="2.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── Globals ──
session = new_session("u2net")
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

DEFAULT_LOGO_PATH = os.path.join(FRONTEND_DIR, "default_logo.png")

# In-memory result store { download_id: { zip: bytes, images: [(name, data), ...] } }
_result_store: dict[str, dict] = {}


def _validate_and_read(file: UploadFile) -> bytes:
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    data = file.file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 10MB)")
    return data


def _save_processed_images(results: list[tuple[str, bytes, float]]) -> str:
    """保存处理结果到内存，返回 download_id"""
    did = uuid.uuid4().hex[:12]
    _result_store[did] = {
        "zip": create_zip(results),
        "images": [(name, data) for name, data, _ in results],
    }
    # 清理旧结果（最多保留 10 份）
    while len(_result_store) > 10:
        _result_store.pop(next(iter(_result_store)))
    return did


# ═══════════════════════════════════════════════════
# V2 路由
# ═══════════════════════════════════════════════════

@app.get("/")
async def root_v2():
    return FileResponse(os.path.join(FRONTEND_DIR, "index_v2.html"))


@app.get("/api/v2/templates")
async def v2_templates():
    return get_templates_list()


@app.get("/api/v2/default_logo/status")
async def v2_default_logo_status():
    """检查是否已有默认LOGO"""
    exists = os.path.exists(DEFAULT_LOGO_PATH)
    if exists:
        size = os.path.getsize(DEFAULT_LOGO_PATH)
        return {"exists": True, "size": size, "filename": "default_logo.png"}
    return {"exists": False}


@app.get("/api/v2/default_logo")
async def v2_default_logo():
    """返回默认LOGO图片"""
    if not os.path.exists(DEFAULT_LOGO_PATH):
        raise HTTPException(404, "未设置默认LOGO")
    with open(DEFAULT_LOGO_PATH, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="image/png")


@app.post("/api/v2/process")
async def v2_process(
    files: list[UploadFile] = File(...),
    logo: UploadFile | None = File(None),
    prompt: str = Form(""),
    template: str = Form(""),
    watermark_position: str = Form(""),
    use_logo: bool = Form(True),
):
    """对话式处理入口。"""
    if not files:
        raise HTTPException(400, "请至少上传一张图片")

    # 解析指令
    if template:
        actions = get_template(template)
        if not actions:
            raise HTTPException(400, f"未知模板: {template}")
    elif prompt:
        parsed = parse_prompt(prompt)
        actions = parsed["actions"]
    else:
        raise HTTPException(400, "请提供指令 (prompt) 或选择模板 (template)")

    if not actions:
        raise HTTPException(400, "无法识别指令，请尝试: 去背景 + 加水印 + 800×800")

    # 读取 logo — 优先使用用户上传的，其次用默认
    logo_bytes = None
    if any(a["action"] == "add_watermark" for a in actions):
        if logo and logo.filename:
            logo_bytes = _validate_and_read(logo)
            # 保存为默认 logo，方便下次使用
            with open(DEFAULT_LOGO_PATH, "wb") as f:
                f.write(logo_bytes)
        elif use_logo and os.path.exists(DEFAULT_LOGO_PATH):
            with open(DEFAULT_LOGO_PATH, "rb") as f:
                logo_bytes = f.read()

    # 水印位置覆盖（前端传参优先于prompt解析）
    if watermark_position:
        for a in actions:
            if a["action"] == "add_watermark":
                a["params"]["position"] = watermark_position

    images: list[tuple[str, bytes]] = []
    for f in files:
        data = _validate_and_read(f)
        images.append((f.filename or "image", data))

    logger.info(f"V2 process: {len(images)} images, actions={actions}")
    t0 = time.perf_counter()
    results = process_batch(images, actions, logo_bytes)
    total_time = time.perf_counter() - t0

    # 存储结果，获取 download_id
    download_id = _save_processed_images(results)

    return JSONResponse({
        "total": len(results),
        "total_time": f"{total_time:.1f}s",
        "actions": actions,
        "download_id": download_id,
        "results": [
            {
                "filename": name,
                "size": len(data),
                "time": f"{t:.1f}s",
                "preview_url": f"/api/v2/preview/{download_id}/{idx}",
            }
            for idx, (name, data, t) in enumerate(results)
        ],
    })


@app.get("/api/v2/preview/{download_id}/{idx}")
async def v2_preview(download_id: str, idx: int):
    """获取单张处理后图片的预览"""
    store = _result_store.get(download_id)
    if not store:
        raise HTTPException(404, "结果已过期，请重新处理")
    images = store["images"]
    if idx < 0 or idx >= len(images):
        raise HTTPException(404, "图片索引不存在")
    name, data = images[idx]
    return Response(content=data, media_type="image/png")


@app.get("/api/v2/download/{download_id}")
async def v2_download(download_id: str):
    """下载 ZIP 打包的处理结果"""
    store = _result_store.get(download_id)
    if not store:
        raise HTTPException(404, "结果已过期，请重新处理")
    zip_bytes = store["zip"]
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="bg-eraser-results-{download_id}.zip"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


# ═══════════════════════════════════════════════════
# 水印检测 + 移除 API
# ═══════════════════════════════════════════════════

@app.post("/api/v2/detect-watermark")
async def v2_detect_watermark(file: UploadFile = File(...)):
    """自动检测图片中的水印区域"""
    data = _validate_and_read(file)
    logger.info(f"detect-watermark: {file.filename}")
    try:
        result = detect_watermarks(data)
        return JSONResponse({
            "filename": file.filename,
            "count": result["count"],
            "regions": result["regions"],
        })
    except Exception as e:
        logger.exception("detect-watermark failed")
        raise HTTPException(500, f"检测失败: {e}")


@app.post("/api/v2/remove-watermark")
async def v2_remove_watermark(
    file: UploadFile = File(...),
    x: int = Form(0),
    y: int = Form(0),
    w: int = Form(0),
    h: int = Form(0),
    auto: bool = Form(True),
    exclude_regions: str = Form(""),
    regions: str = Form(""),
):
    """去除水印

    - auto=true: 自动检测并去除所有水印
    - auto=false: 使用 x,y,w,h 指定的矩形区域
    - exclude_regions: JSON 字符串，要排除的区域 [{x,y,w,h},...]
    - regions: JSON 字符串，用户调整后的精确区域 [{x,y,w,h},...]（优先级最高，跳过自动检测）
    """
    data = _validate_and_read(file)
    logger.info(f"remove-watermark: {file.filename}, auto={auto}")

    # 如果前端传了精确区域（用户调整后的坐标），直接用它们
    user_regions = []
    if regions:
        try:
            raw = json.loads(regions)
            # 确保坐标为整数（前端拖拽可能产生浮点数）
            user_regions = [
                {k: int(v) for k, v in r.items() if k in ('x', 'y', 'w', 'h')}
                for r in raw
            ]
        except json.JSONDecodeError:
            pass

    exclude = []
    if exclude_regions:
        try:
            exclude = json.loads(exclude_regions)
        except json.JSONDecodeError:
            pass

    try:
        if user_regions:
            # 用户手动调整过的精确区域，直接擦除
            result_bytes = _erase_watermarks_batch(data, user_regions)
            regions_out = user_regions
        elif auto:
            result = remove_watermarks_auto(data, exclude_regions=exclude)
            if not result["success"]:
                return JSONResponse({
                    "success": False,
                    "message": "未检测到水印区域，请手动框选",
                    "preview_url": None,
                })
            result_bytes = result["result_bytes"]
            regions_out = result["regions"]
        else:
            if w <= 0 or h <= 0:
                raise HTTPException(400, "请指定有效的水印区域 (x,y,w,h)")
            result_bytes = remove_watermark_by_rect(data, x, y, w, h)
            regions_out = [{"x": x, "y": y, "w": w, "h": h, "method": "manual"}]

        # 保存到临时预览
        did = uuid.uuid4().hex[:12]
        from backend.processor.engine import create_zip
        _result_store[did] = {
            "zip": create_zip([("watermark_removed.png", result_bytes, 0.0)]),
            "images": [("watermark_removed.png", result_bytes)],
        }

        return JSONResponse({
            "success": True,
            "message": f"已去除 {len(regions_out)} 处水印",
            "regions": regions_out,
            "preview_url": f"/api/v2/preview/{did}/0",
            "download_id": did,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("remove-watermark failed")
        raise HTTPException(500, f"去除水印失败: {e}")


@app.post("/api/v2/search-watermark-text")
async def v2_search_watermark_text(
    file: UploadFile = File(...),
    text: str = Form(...),
):
    """根据输入的文字内容搜索图片中的水印位置"""
    data = _validate_and_read(file)
    logger.info(f"search-watermark-text: {file.filename}, text={text}")

    try:
        result = find_watermark_by_text(data, text)
        return JSONResponse(result)
    except Exception as e:
        logger.exception("search-watermark-text failed")
        raise HTTPException(500, f"搜索失败: {e}")


# ═══════════════════════════════════════════════════
# 商品渲染 API
# ═══════════════════════════════════════════════════

render_jobs = {}  # task_id → 最后一次轮询到的状态缓存


@app.post("/api/v2/render/analyze")
async def render_analyze(
    file: Optional[UploadFile] = File(None),
    prompt: str = Form(...),
    width: int = Form(1024),
    height: int = Form(1024),
):
    """分析参考图（可选）并提交渲染任务

    返回 task_id + 分析结果，前端轮询 /render/status 获取进度
    """
    from backend.processor.engine import create_zip

    data = _validate_and_read(file) if file else None
    logger.info(f"render analyze: {file.filename if file else '无参考图'}, prompt='{prompt[:50]}...'")

    try:
        result = analyze_and_render(
            image_bytes=data,
            prompt=prompt,
            width=width,
            height=height,
        )
        resp = {
            "task_id": result["id"],
            "status": result["status"],
            "analysis": result.get("analysis", {}),
            "scene_prompt": result.get("scene_prompt", prompt),
            "total_time": _fmt_time(result.get("updated_at", 0) - result.get("created_at", 0)),
        }
        # 如果同步完成，直接提供下载
        if result.get("result_bytes") and result["status"] == "completed":
            did = uuid.uuid4().hex[:12]
            _result_store[did] = {
                "zip": create_zip([("render_result.png", result["result_bytes"], 0.0)]),
                "images": [("render_result.png", result["result_bytes"])],
            }
            resp["preview_url"] = f"/api/v2/preview/{did}/0"
            resp["download_id"] = did
        return JSONResponse(resp)
    except Exception as e:
        logger.exception("render analyze failed")
        raise HTTPException(500, f"渲染分析失败: {e}")


@app.get("/api/v2/render/status/{task_id}")
async def render_status(task_id: str):
    """轮询渲染任务状态"""
    result = poll_render_status(task_id)
    if result.get("status") == "not_found":
        raise HTTPException(404, "任务不存在")

    resp = {
        "task_id": task_id,
        "status": result["status"],
        "scene_prompt": result.get("scene_prompt", ""),
        "progress": result.get("progress", 0),
        "error": result.get("error"),
        "total_time": _fmt_time(result.get("elapsed", 0)),
    }

    if result.get("result_bytes"):
        # 缓存结果
        did = uuid.uuid4().hex[:12]
        from backend.processor.engine import create_zip
        _result_store[did] = {
            "zip": create_zip([("render_result.png", result["result_bytes"], 0.0)]),
            "images": [("render_result.png", result["result_bytes"])],
        }
        resp["preview_url"] = f"/api/v2/preview/{did}/0"
        resp["download_id"] = did

    if result.get("result_url"):
        resp["result_url"] = result["result_url"]

    return JSONResponse(resp)


@app.get("/api/v2/render/jobs")
async def render_jobs_list():
    """获取所有渲染任务列表"""
    jobs = task_store.get_all()
    return JSONResponse([{
        "id": j["id"],
        "status": j["status"],
        "prompt": j.get("scene_prompt", j["prompt"])[:80],
        "size": j["size"],
        "error": j.get("error"),
    } for j in jobs[-20:]])  # 最近 20 条


# ═══════════════════════════════════════════════════
# V1 路由（经典版）
# ═══════════════════════════════════════════════════

@app.get("/classic")
@app.get("/classic/")
async def root_classic():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    data = _validate_and_read(file)
    logger.info(f"remove_bg: {file.filename}")
    try:
        output = remove(data, session=session)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    name = os.path.splitext(file.filename or "image")[0]
    return Response(
        content=output, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}_nobg.png"'},
    )


@app.post("/remove-bg-with-color")
async def remove_bg_with_color(file: UploadFile = File(...), color: str = Form("#ffffff")):
    data = _validate_and_read(file)
    try:
        output = engine_change_bg(data, color)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    c = color.lstrip("#")
    name = os.path.splitext(file.filename or "image")[0]
    return Response(
        content=output, media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{name}_bg{c}.png"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
