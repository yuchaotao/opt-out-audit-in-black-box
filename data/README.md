# Data

## RetailRocket (real interactions)

Source: Retailrocket recommender system dataset, https://www.kaggle.com/retailrocket/ecommerce-dataset
(`events.csv`: 2,756,101 view / add-to-cart / transaction events, 139 days).

We do **not** redistribute the raw file, because we could not verify its license from a primary
source. `fetch_retailrocket.sh` downloads an identical public copy and prints its checksum
(first 16 hex digits of SHA-256: `3745aa83238b1e6d`). Alternatively, download it from Kaggle into
`data/retailrocket/events.csv`.

`sim/real_data.py` builds the audit snapshots on first use and caches them in
`data/retailrocket/processed.npz` (also not committed).

Experiment outputs on this data are in `results/real/`.
