#!/bin/bash
# P2: queue every ablation row (start before sleeping). Fast rows first, so results exist even if the night is cut short.
# Usage: N=300 bash eval/run_all.sh 2>&1 | tee results/run_all.log
for c in bk_dpm20_taesd bk_dpm20 bk_fp16_50 sd14_fp16_50 sd14_fp32_50 bk_final; do
  if [ -f "results/${c}_gpu/metrics.json" ]; then echo "skip $c (done)"; continue; fi
  python eval/run_eval.py --config $c --n ${N:-300} || echo "FAILED $c"
done
python eval/plot_results.py
