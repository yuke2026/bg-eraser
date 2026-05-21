# BgEraser 🎨

AI 智能去背景工具 — **一键上传，秒出结果**。

## 快速开始

```bash
# 1. 启动后端
chmod +x start.sh && ./start.sh

# 2. 另一个终端，启动前端
cd frontend && python3 -m http.server 8080

# 3. 浏览器打开 http://localhost:8080
```

## 项目结构

```
bg-eraser/
├── backend/
│   ├── main.py              # FastAPI 后端 (去背景 API)
│   └── requirements.txt     # Python 依赖
├── frontend/
│   └── index.html           # 前端页面 (自包含)
├── start.sh                 # 一键启动脚本
└── README.md
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/remove-bg` | 上传图片，返回透明 PNG |
| POST | `/remove-bg-with-color` | 上传图片 + 指定颜色，返回带色 PNG |
| GET  | `/health` | 健康检查 |

### 前端功能

- ✅ 拖拽 / 点击上传图片
- ✅ AI 去背景（rembg U²-Net 模型）
- ✅ 原图/结果对比滑块
- ✅ 三种视图模式：对比 / 原图 / 结果
- ✅ 换背景色（8 种预设 + 取色器）
- ✅ 一键下载 PNG
- ✅ 错误提示 + 重试

## 技术栈

- **前端**: HTML + TailwindCSS (CDN) + vanilla JS
- **后端**: Python + FastAPI + rembg
- **模型**: U²-Net (rembg)
