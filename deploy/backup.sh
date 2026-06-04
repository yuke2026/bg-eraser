#!/bin/bash
# ============================================================
# PicMagic (bg-eraser) 自动备份脚本
# 功能：代码Git推送 + 配置文件存档
# 位置：deploy/backup.sh
# 建议：每周运行一次（cron）
# ============================================================
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$BASE_DIR/deploy/backups"
DATE_STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$BASE_DIR/deploy/backup.log"

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== PicMagic 备份开始 ==="

# ── 1. 备份非Git追踪的关键文件 ──
log "[1/4] 备份 .env 配置文件..."
if [ -f "$BASE_DIR/.env" ]; then
    cp "$BASE_DIR/.env" "$BACKUP_DIR/.env.$DATE_STAMP"
    log "  ✔ .env → $BACKUP_DIR/.env.$DATE_STAMP"
fi

log "[2/4] 备份用户LOGO..."
if [ -f "$BASE_DIR/frontend/default_logo.png" ]; then
    cp "$BASE_DIR/frontend/default_logo.png" "$BACKUP_DIR/default_logo.$DATE_STAMP.png"
    log "  ✔ default_logo.png → $BACKUP_DIR/default_logo.$DATE_STAMP.png"
fi

# ── 3. Git 状态检查 & 推送 ──
log "[3/4] 检查 Git 状态..."
cd "$BASE_DIR"

UNCOMMITTED=$(git status --porcelain 2>/dev/null | wc -l)
if [ "$UNCOMMITTED" -gt 0 ]; then
    log "  ⚠ 发现 $UNCOMMITTED 个未提交的变更:"
    git status --short | tee -a "$LOG_FILE"
    log "  → 请手动 commit 这些变更: git add . && git commit -m '...'"
else
    log "  ✔ 工作区干净"
fi

log "[4/4] 推送已提交代码到 GitHub..."
if git push origin main 2>&1 | tee -a "$LOG_FILE"; then
    log "  ✔ GitHub 推送成功"
else
    log "  ⚠ GitHub 推送失败（网络或SSH问题）"
fi

# ── 清理：保留最近30天的备份 ──
find "$BACKUP_DIR" -name ".env.*" -mtime +30 -delete 2>/dev/null
find "$BACKUP_DIR" -name "default_logo.*" -mtime +30 -delete 2>/dev/null

log "=== PicMagic 备份完成 ==="
echo ""
