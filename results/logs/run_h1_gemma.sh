#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h1_gemma.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h1_gemma run >> $LOG 2>&1
echo "run exit $? $(date)" >> $LOG
F=results/trust_score_heldout_gemma-3-12b.json
if [ ! -f $F ]; then echo "final result missing" >> $LOG; fi
echo "ALL DONE $(date)" >> $LOG
