#!/usr/bin/env bash
# Compile the paper, regenerate README page images, and refresh the image list in README.md.
# Needs latexmk, poppler-utils (pdftoppm) and python3.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -quiet main.tex
rm -f pages/page-*.png
mkdir -p pages
pdftoppm -png -r 110 main.pdf pages/page
python3 - <<'PY'
import os, re
pages = sorted(f for f in os.listdir("pages") if f.endswith(".png"))
imgs = "\n".join(f'<img src="pages/{f}" width="100%" alt="Page {i}">' for i, f in enumerate(pages, 1))
r = open("README.md").read()
r = re.sub(r'(<img src="pages/[^\n]*\n?)+', imgs + "\n", r)
open("README.md", "w").write(r)
print(f"Built main.pdf and {len(pages)} page images; README updated.")
PY
