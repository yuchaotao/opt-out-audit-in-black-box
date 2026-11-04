#!/usr/bin/env bash
# Compile the paper and regenerate README page images (needs latexmk and poppler-utils).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -quiet main.tex
rm -f pages/page-*.png
mkdir -p pages
pdftoppm -png -r 110 main.pdf pages/page
echo "Built main.pdf and $(ls pages | wc -l) page images. Update the image list in README.md if the page count changed."
