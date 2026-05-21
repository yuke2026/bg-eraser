"""BgEraser — V1 经典版 + V2 专业版"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rembg import remove, new_session
from PIL import Image
import io, os, logging, time, uuid

from backend.nlu.parser import parse_prompt, get_templates_list, get_template
from backend.processor.engine import (
    process_batch, create_zip,
    remove_background, change_bg_color as engine_change_bg,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bg-eraser")

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
