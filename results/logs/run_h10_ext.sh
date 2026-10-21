#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h10_extended.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h10_extended >> $LOG 2>&1; echo "run exit $? $(date)" >> $LOG
.venv/bin/python -m hlens.experiments.h10_extended --analyze > results/logs/h10_extended_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
