#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h10_control.log
echo "START $(date)" >> $LOG
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h10_control"
$P run >> $LOG 2>&1; echo "run exit $? $(date)" >> $LOG
$P analyze > results/logs/h10_control_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
