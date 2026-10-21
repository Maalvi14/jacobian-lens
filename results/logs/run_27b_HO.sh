#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h9_27b_HO.log
echo "START $(date)" >> $LOG
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h9_replication --clean --model qwen3.6-27b --set HO >> $LOG 2>&1
echo "EXIT CODE $? at $(date)" >> $LOG
