#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/e1_heldout.log
echo "START $(date)" >> $LOG
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.heldout_h1_e1"
$P e1 >> $LOG 2>&1; echo "e1 exit $? $(date)" >> $LOG
$P analyze > results/logs/e1_heldout_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
