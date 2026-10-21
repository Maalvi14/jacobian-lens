#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h9_27b_O.log
echo "START $(date)" >> $LOG
env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h9_replication --clean --model qwen3.6-27b --set O >> $LOG 2>&1
echo "EXIT CODE $? at $(date)" >> $LOG
F=results/h9_replication/O_qwen3.6-27b_clean_mixed.json
if [ ! -f $F ]; then
  echo "final result missing; analyzing the checkpoint" >> $LOG
  .venv/bin/python -m hlens.experiments.h9_replication --clean --model qwen3.6-27b --set O --from-checkpoint >> $LOG 2>&1
fi
echo "FINISHED $(date)" >> $LOG
