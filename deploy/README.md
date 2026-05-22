# BgEraser 部署指南

## systemd 服务（推荐）

```bash
# 安装
sudo cp deploy/bg-eraser.service /etc/systemd/system/
sudo systemctl daemon-reload

# 启动
sudo systemctl start bg-eraser

# 开机自启
sudo systemctl enable bg-eraser

# 查看状态
sudo systemctl status bg-eraser

# 查看日志
sudo journalctl -u bg-eraser -f

# 重启
sudo systemctl restart bg-eraser

# 停止
sudo systemctl stop bg-eraser
```

## 访问地址

```
http://服务器IP:8000/
```

## 注意事项

- 需要先安装 Python 依赖（见 `backend/requirements.txt`）
- 确保服务器防火墙放行了 8000 端口
- 如果需要绑定域名，建议用 Nginx 反向代理到 8000 端口
