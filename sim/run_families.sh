#!/bin/bash
# Feature-family experiments (Sec. 7): synthetic and RetailRocket semi-synthetic. At most 2 processes.
set -e
cd "$(dirname "$0")"
R=../results/families; mkdir -p $R
syn() {
  python3 families.py synthetic 100,101,102,103,104,105,106,107,108,109 $R/syn_clean.json clean 0 user 10
  for rho in 0.1 1.0; do python3 families.py synthetic 0,1,2 $R/syn_leak_user_$rho.json leak $rho user 10; done
  python3 naive_single.py synthetic $(seq -s, 200 229) $R/naive_single_syn.json 10
}
real() {
  python3 families.py real 0,1,2 $R/real_clean.json clean 0 user 7
  for rho in 0.1 1.0; do python3 families.py real 0,1 $R/real_leak_user_$rho.json leak $rho user 7; done
  python3 naive_single.py real 0,1,2,3 $R/naive_single_real.json 7
}
case "$1" in syn) syn;; real) real;; *) syn & real; wait;; esac
