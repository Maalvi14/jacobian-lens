#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/bprime_llama.log
M="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h2_replication"
C="--config B3 --model llama3.1-8b"
echo "START $(date)" >> $LOG
for s in 20 21 22; do $M null $C --seed $s --layers 16 >> $LOG 2>&1; echo "null s$s L16 exit $? $(date)" >> $LOG; done
$M null $C --seed 20 --layers 8 24 >> $LOG 2>&1; echo "null s20 L8-24 exit $? $(date)" >> $LOG
$M real $C >> $LOG 2>&1; echo "real exit $? $(date)" >> $LOG
.venv/bin/python -m hlens.experiments.h2_replication analyze $C > results/logs/bprime_llama_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
