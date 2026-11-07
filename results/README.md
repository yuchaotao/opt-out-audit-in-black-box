# Results

All JSON files hold daily p-values per test (keys `T1`...`T7`, naive tests `C1_*`...`C6_*`), per seed and configuration.

- `naive_summary.json`: false-alarm rates of naive tests and fixes (Table 5).
- `synthetic/`: simulator runs (calibration, per-edge and per-customer leak sweeps, T7, naive tests,
  refined T5/T6). `summary.json` is produced by `sim/analyze.py`.
- `real/`: RetailRocket runs: basic tests on clean pipelines (`real_null_basic.json`), basic vs refined
  T5/T6 (`rob_real_*.json`), T1/T2/T4 power (`real_edge.json`), naive tests (`naive_real.json`).
- `families/`: each feature family of Table 2 (A1-A3, R1-R4) tested separately (Table "Feature families"):
  `syn_*.json`, `real_*.json` hold daily p-values per `<family>_<test>` (`R2_T2n` = T2 without the 3-hop
  churn adjustment); `naive_single_*.json` hold the single-node naive test (Pitfalls 1-3);
  `summary.json` and `table.tex` come from `sim/summarize_families.py` and `sim/make_family_table.py`.
