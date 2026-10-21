#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h9_scan_gemma.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h9_scan --model gemma-3-12b >> $LOG 2>&1; echo "scan exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
