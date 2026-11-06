#!/usr/bin/env bash
# Real-data (RetailRocket) experiments. Run ../data/fetch_retailrocket.sh first.
# Each real-data process holds the 400k-customer graph; run at most two at a time on 8 GB RAM.
set -euo pipefail
cd "$(dirname "$0")"
python3 run_real.py edge 100,101,102 real_null_basic.json null 14         # T1 T2 T4 + basic T5/T6, clean
python3 eval_robust.py real clean 0 edge 200,201,202,203 rob_real_null.json 14   # basic vs refined T5/T6, clean
for pat in edge user; do for rho in 0.1 1.0; do
  python3 eval_robust.py real global $rho $pat 0,1 rob_real_${pat}_${rho}.json 10
done; done
python3 run_real.py edge 0,1 real_edge.json sweep 10                       # T1 T2 T4 power, per-edge leaks
python3 naive.py real 300,301,302,303 naive_real.json 10                   # naive tests C1-C6 on real data
