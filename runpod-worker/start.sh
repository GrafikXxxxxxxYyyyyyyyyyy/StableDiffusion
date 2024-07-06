#!/bin/bash
echo "Worker Initiated"
CURRENT_DIR=$(pwd)

echo "Starting RunPod Handler"
python3.11 -u ${CURRENT_DIR}/runpod-worker/rp_handler.py
