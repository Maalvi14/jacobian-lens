#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h6_h8_gemma.log
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m"
echo "START $(date)" >> $LOG
$P hlens.experiments.h6_replication --model gemma-3-12b --arm null >> $LOG 2>&1; echo "H6 null exit $? $(date)" >> $LOG
$P hlens.experiments.h6_replication --model gemma-3-12b --arm real >> $LOG 2>&1; echo "H6 real exit $? $(date)" >> $LOG
$P hlens.experiments.h6_replication --model gemma-3-12b --arm analyze >> results/logs/h6_gemma_analysis.log 2>&1; echo "H6 analyze exit $? $(date)" >> $LOG
$P hlens.experiments.h6_replication --model gemma-3-12b --arm knee >> $LOG 2>&1; echo "H6 knee exit $? $(date)" >> $LOG
$P hlens.experiments.h8_replication --model gemma-3-12b >> results/logs/h8_gemma.log 2>&1; echo "H8 exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
