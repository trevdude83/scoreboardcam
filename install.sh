#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/rocketsessions-scoreboardcam"
SERVICE_PATH="/etc/systemd/system/scoreboardcam.service"

sudo mkdir -p "$APP_DIR"
sudo rsync -av --exclude ".venv" --exclude "__pycache__" ./ "$APP_DIR/"

cd "$APP_DIR"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

sudo cp service/scoreboardcam.service "$SERVICE_PATH"
sudo systemctl daemon-reload
sudo systemctl enable scoreboardcam.service
sudo systemctl restart scoreboardcam.service

printf "ScoreboardCam service installed and started.\n"
