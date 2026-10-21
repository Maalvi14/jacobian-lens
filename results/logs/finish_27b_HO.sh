#!/bin/bash
# after the run exits: if it did not write the final result, build it from the checkpoint
cd /home/mv/repos/jacobian-lens
until grep -q "EXIT CODE" results/logs/h9_27b_HO.log 2>/dev/null; do sleep 30; done
F=results/h9_replication/HO_qwen3.6-27b_clean_mixed.json
if [ ! -f $F ]; then
  echo "final result missing; analyzing the checkpoint" >> results/logs/h9_27b_HO.log
  .venv/bin/python -m hlens.experiments.h9_replication --clean --model qwen3.6-27b --set HO --from-checkpoint >> results/logs/h9_27b_HO.log 2>&1
fi
echo "FINISHED $(date)" >> results/logs/h9_27b_HO.log
