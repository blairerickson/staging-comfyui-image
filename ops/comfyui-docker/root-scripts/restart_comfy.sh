#!/bin/bash
tmux kill-session -t comfy 2>/dev/null; pkill -f "python main.py --listen 0.0.0.0 --port 8188"; sleep 5
tmux new-session -d -s comfy "cd /root/ComfyUI && DEEPSEEK_API_KEY= /root/venv/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-manager 2>&1 | tee /root/boot_logs"
