#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/swap_foldcase.log
P=".venv/bin/python -X faulthandler -u -m hlens.experiments.swap_heldout --fold-case"
echo "START $(date)" >> $LOG
$P --model qwen3.5-4b >> $LOG 2>&1; echo "qwen exit $? $(date)" >> $LOG
$P --model gemma-3-12b >> $LOG 2>&1; echo "gemma exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
