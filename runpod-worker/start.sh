#!/bin/bash
echo "Worker Initiated"
CURRENT_DIR=$(pwd)

# echo "W&B login"
# wandb login ffda53cd1b029f35763d3353ff8c89730c81a05d

echo "Starting RunPod Handler"
python3.11 -u ${CURRENT_DIR}/runpod-worker/rp_handler.py
