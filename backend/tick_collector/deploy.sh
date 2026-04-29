#!/bin/bash
# Deploy tick collector to VPS as standalone service
set -e

VPS="vps"
REMOTE_DIR="/home/linuxuser/tick_collector"

echo "=== Deploying Tick Collector to VPS ==="

# Create remote directory
ssh $VPS "mkdir -p $REMOTE_DIR"

# Sync files
rsync -avz --exclude='data/' --exclude='venv/' --exclude='__pycache__/' \
    ./ $VPS:$REMOTE_DIR/

# Setup venv + install deps
ssh $VPS "cd $REMOTE_DIR && python3 -m venv venv && source venv/bin/activate && pip install -q -r requirements.txt"

# Install systemd service
ssh $VPS "sudo cp $REMOTE_DIR/tick-collector.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable tick-collector"

echo ""
echo "=== Deployed! ==="
echo "Start:   ssh $VPS 'sudo systemctl start tick-collector'"
echo "Status:  ssh $VPS 'sudo systemctl status tick-collector'"
echo "Logs:    ssh $VPS 'journalctl -u tick-collector -f'"
echo "Data:    ssh $VPS 'ls -la $REMOTE_DIR/data/'"
