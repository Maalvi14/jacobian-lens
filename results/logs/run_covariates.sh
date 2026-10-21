#!/bin/bash
cd /home/mv/repos/jacobian-lens
LOG=results/logs/h10_covariates.log
echo "START $(date)" >> $LOG
P="env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -X faulthandler -u -m hlens.experiments.h10_covariates"
$P compute >> $LOG 2>&1; echo "compute exit $? $(date)" >> $LOG
$P analyze > results/logs/h10_covariates_analysis.log 2>&1; echo "analyze exit $? $(date)" >> $LOG
echo "ALL DONE $(date)" >> $LOG
