#!/bin/bash
# start the 27B held-out run once >= 105 GiB are available (after the user drops the page cache)
until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 105 ]; do sleep 10; done
exec /home/mv/repos/jacobian-lens/results/logs/run_27b_HO.sh
