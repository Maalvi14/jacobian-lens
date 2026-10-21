#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h1_heldout.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.heldout_h1_e1 h1 >> $LOG 2>&1; echo "h1 exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
