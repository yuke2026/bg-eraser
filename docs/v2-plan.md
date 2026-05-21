---

## 十一、版本共存方案

### 路由设计

```
http://server:8000/           → V2 专业版（电商AI作图工具）⭐ 默认页
http://server:8000/classic/   → V1 经典版（简单去背景工具）

API 接口:
POST /remove-bg              → 去背景 (V1 & V2 共用)
POST /remove-bg-with-color   → 去背景+换色 (V1 & V2 共用)
POST /api/v2/process         → V2 对话式处理 (新增)
GET  /api/v2/presets         → V2 尺寸预设 (新增)
```

### V2 UI 头部导航（在 header 右侧加切换按钮）

```
┌─────────────────────────────────────────────────────┐
│  PicMagic ✨           [📖教程] [🎯经典版]          │
├──────────┬──────────┬───────────────────────────────┤
│  ① 上传  │  ② 指令  │  ③ 结果                       │
│  ...     │  ...     │  ...                          │
└──────────┴──────────┴───────────────────────────────┘
```

### V1 UI 头部导航（加跳转到 V2 的入口）

```
┌────────────────────────────────────────────────┐
│  BgEraser            [🚀 切换到专业版]          │
│  免费 AI 去背景                                 │
└────────────────────────────────────────────────┘
```

### 后端实现

```python
# V2 首页
@app.get("/")
async def root_v2():
    return FileResponse(os.path.join(FRONTEND_DIR, "index_v2.html"))

# V1 经典版
@app.get("/classic")
@app.get("/classic/")
async def root_classic():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
```

### 前端文件结构

```
frontend/
├── index.html          ← V1 经典版 (现有的)
├── index_v2.html       ← V2 专业版 (新建)
└── (v2/)              ← V2 需要的子资源
```

### 迁移策略

1. 复制 `index.html` → 备份为保留的 V1
2. 新建 `index_v2.html` → V2 UI
3. 修改 `index.html` header → 添加 "切换到专业版" 按钮
4. 修改 backend main.py → 根路径指向 V2，`/classic` 指向 V1
5. V1 的 API 接口保持不变，V2 共用

---

## 十二、2025-05-21 五项用户体验优化

### 1. 水印位置选择器
- **新增前端 UI**：尺寸预设下方添加 "📍 水印位置" 四个按钮（左上/右上/右下/左下）
- **默认位置**：左上角（`top-left`）
- **实现**：位置通过 `watermark_position` 表单字段独立传递，后端覆盖 prompt 解析结果
- **涉及文件**：`frontend/index_v2.html`（HTML+JS），`backend/main.py`（新增参数）

### 2. 水印大小适中（约75%图片宽度）
- **后端**：`engine.py` 中 `size_ratio` 默认值从 `0.15` → `0.75`
- **效果**：LOGO 宽度约占图片宽度的 75%，对所有图片类型通用

### 3. 尺寸 contain 模式（不裁剪不拉伸）
- **所有模板**：淘宝、京东、拼多多、抖音 → resize mode 都改为 `contain`
- **自定义提示**：`process_pipeline` 中 resize 默认 mode 从 `"cover"` → `"contain"`
- **涉及文件**：`backend/nlu/parser.py`（模板），`backend/processor/engine.py`（默认值）

### 4. 默认 LOGO 持久化
- **后端新增端点**：
  - `GET /api/v2/default_logo/status` → 检查默认 LOGO 是否存在
  - `GET /api/v2/default_logo` → 返回默认 LOGO 图片
- **前端改进**：
  - 页面加载时自动检测默认 LOGO，显示 "默认LOGO ✓"
  - 上传新 LOGO 可覆盖，删除后自动回退到默认
  - `use_logo` 表单字段控制是否使用默认 LOGO
- **涉及文件**：`backend/main.py`（新端点），`frontend/index_v2.html`（自动加载）

### 5. 快捷指令更新
- 指令文本从 "右下角" → 不包含位置，由位置选择器决定
- 保持 "去背景+水印"、"仅加水印" 等简洁指令名

