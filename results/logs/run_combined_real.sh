#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/combined_real_model.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m pytest -m real_model -v --durations=0 -p no:cacheprovider >> $LOG 2>&1
echo "pytest exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
