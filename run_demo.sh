#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Starting MuJoCo Demo on DISPLAY=${DISPLAY} ..."
python3 interactive_demo.py
