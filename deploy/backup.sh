#!/bin/bash
# ============================================================
# PicMagic (bg-eraser) 自动备份脚本（智能跳过版）
# 功能：仅在有实际变更时执行备份，避免浪费资源
# 位置：deploy/backup.sh
# ============================================================
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$BASE_DIR/deploy/backups"
DATE_STAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$BASE_DIR/deploy/backup.log"
LAST_RUN_FILE="$BACKUP_DIR/.last_backup"

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== PicMagic 备份开始 ==="

# ── 0. 变更检测 — 没有变更就跳过，不浪费资源 ──
cd "$BASE_DIR"

# 获取源头：最近一次 git commit 时间、.env mtime、LOGO mtime
LATEST_COMMIT=$(git log -1 --format=%ct origin/main 2>/dev/null || echo 0)
ENV_MTIME=$(stat -c %Y "$BASE_DIR/.env" 2>/dev/null || echo 0)
LOGO_MTIME=$(stat -c %Y "$BASE_DIR/frontend/default_logo.png" 2>/dev/null || echo 0)
UNCOMMITTED_COUNT=$(git status --porcelain 2>/dev/null | wc -l)

# 三件事中最新的一条
LATEST_CHANGE=$LATEST_COMMIT
[ "$ENV_MTIME" -gt "$LATEST_CHANGE" ] && LATEST_CHANGE=$ENV_MTIME
[ "$LOGO_MTIME" -gt "$LATEST_CHANGE" ] && LATEST_CHANGE=$LOGO_MTIME
[ "$UNCOMMITTED_COUNT" -gt 0 ] && LATEST_CHANGE=$(date +%s)  # 有未提交变更 → 视为现在

# 跟上次备份比较
if [ -f "$LAST_RUN_FILE" ]; then
    LAST_TIME=$(cat "$LAST_RUN_FILE")
    if [ "$LATEST_CHANGE" -le "$LAST_TIME" ] 2>/dev/null; then
        log "🟢 跳过备份：自上次备份 ($(date -d @$LAST_TIME '+%Y-%m-%d %H:%M')) 以来无任何变更"
        echo ""
        exit 0
    fi
fi

# ── 1. 备份 .env（仅当有变更） ──
log "[1/4] 检测 .env 变更..."
if [ -f "$BASE_DIR/.env" ]; then
    LAST_ENV_BACKUP=$(ls -t "$BACKUP_DIR"/.env.* 2>/dev/null | head -1)
    if [ -n "$LAST_ENV_BACKUP" ]; then
        if ! cmp -s "$BASE_DIR/.env" "$LAST_ENV_BACKUP"; then
            cp "$BASE_DIR/.env" "$BACKUP_DIR/.env.$DATE_STAMP"
            log "  ✔ .env 有变更 → 已备份"
        else
            log "  ➖ .env 无变更 → 跳过"
        fi
    else
        cp "$BASE_DIR/.env" "$BACKUP_DIR/.env.$DATE_STAMP"
        log "  ✔ .env 首次备份"
    fi
fi

# ── 2. 备份 LOGO（仅当有变更） ──
log "[2/4] 检测 LOGO 变更..."
if [ -f "$BASE_DIR/frontend/default_logo.png" ]; then
    LAST_LOGO_BACKUP=$(ls -t "$BACKUP_DIR"/default_logo.*.png 2>/dev/null | head -1)
    if [ -n "$LAST_LOGO_BACKUP" ]; then
        if ! cmp -s "$BASE_DIR/frontend/default_logo.png" "$LAST_LOGO_BACKUP"; then
            cp "$BASE_DIR/frontend/default_logo.png" "$BACKUP_DIR/default_logo.$DATE_STAMP.png"
            log "  ✔ LOGO 有变更 → 已备份"
        else
            log "  ➖ LOGO 无变更 → 跳过"
        fi
    else
        cp "$BASE_DIR/frontend/default_logo.png" "$BACKUP_DIR/default_logo.$DATE_STAMP.png"
        log "  ✔ LOGO 首次备份"
    fi
fi

# ── 3. Git 状态检查 ──
log "[3/4] 检查 Git 状态..."
if [ "$UNCOMMITTED_COUNT" -gt 0 ]; then
    log "  ⚠ 发现 $UNCOMMITTED_COUNT 个未提交的变更:"
    git status --short | tee -a "$LOG_FILE"
    log "  → 请手动 commit: git add . && git commit -m '...'"
else
    log "  ✔ 工作区干净"
fi

# ── 4. 推送 — 只有新提交才推送 ──
log "[4/4] 检查是否需要推送..."
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
if [ "$AHEAD" -gt 0 ]; then
    log "  📤 $AHEAD 个新提交待推送..."
    if git push origin main 2>&1 | tee -a "$LOG_FILE"; then
        log "  ✔ GitHub 推送成功"
    else
        log "  ⚠ GitHub 推送失败（网络或SSH问题）"
    fi
else
    log "  ➖ 无新提交 → 跳过推送"
fi

# ── 记录本次备份时间 ──
date +%s > "$LAST_RUN_FILE"

# ── 清理：保留最近30天的备份 ──
find "$BACKUP_DIR" -name ".env.*" -mtime +30 -delete 2>/dev/null
find "$BACKUP_DIR" -name "default_logo.*" -mtime +30 -delete 2>/dev/null

log "=== PicMagic 备份完成 ==="
echo ""
