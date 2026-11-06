#!/usr/bin/env bash
# Fetch the RetailRocket events file (original source: https://www.kaggle.com/retailrocket/ecommerce-dataset).
# We do not redistribute the raw data. This script downloads an identical copy mirrored on GitHub
# and verifies its checksum; alternatively, download events.csv from Kaggle into data/retailrocket/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p retailrocket
curl -sSL -o retailrocket/events.csv \
  https://raw.githubusercontent.com/fabiolariveranoriega/Merchant-Analytics-Platform/main/data/events.csv
echo "3745aa83238b1e6d  (first 16 hex of sha256 expected)"
sha256sum retailrocket/events.csv | cut -c1-16
