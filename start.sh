#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "╔════════════════════════════════════════╗"
echo "║        BgEraser - 启动脚本             ║"
echo "╚════════════════════════════════════════╝"
echo ""

# ── Check Python ──
if ! command -v python3 &>/dev/null; then
  echo "❌ 请先安装 Python 3.10+"
  exit 1
fi

# ── Create virtualenv if needed ──
VENV="$ROOT/backend/.venv"
if [ ! -d "$VENV" ]; then
  echo "📦 创建虚拟环境..."
  python3 -m venv "$VENV"
fi

# ── Install dependencies ──
echo "📥 安装后端依赖..."
"$VENV/bin/pip" install -q -r "$ROOT/backend/requirements.txt" 2>&1 | tail -1

echo ""
echo "✅ 后端启动: http://localhost:8000"
echo "🌐 前端地址: 直接用浏览器打开"
echo "     file://$ROOT/frontend/index.html"
echo ""
echo "或者用 python3 -m http.server 托管前端:"
echo "  cd frontend && python3 -m http.server 8080"
echo "  然后访问 http://localhost:8080"
echo ""

# ── Start backend ──
cd "$ROOT/backend"
exec "$VENV/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 --reload
