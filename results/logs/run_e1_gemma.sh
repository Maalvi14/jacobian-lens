#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/e1_gemma.log
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.e1_gemma"
echo "START $(date)" >> $LOG
$P real >> $LOG 2>&1; echo "real exit $? $(date)" >> $LOG
$P null >> $LOG 2>&1; echo "null exit $? $(date)" >> $LOG
$P analyze > results/logs/e1_gemma_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
