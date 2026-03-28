#!/bin/bash

set -e

cd "$(dirname "$0")"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

choose_python() {
  for candidate in ".venv/bin/python" "../mth8302-devoir2/.venv/bin/python"; do
    if [ -x "$candidate" ] && "$candidate" -c "import numpy, pandas, scipy, sklearn, statsmodels, matplotlib" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(choose_python || true)"

if [ -n "$PYTHON_BIN" ]; then
  "$PYTHON_BIN" src/main.py
else
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
  mkdir -p "$UV_CACHE_DIR"
  uv run src/main.py
fi

echo "All scripts completed."

echo "Compiling LaTeX..."
cd report
TEX_NAME="MTH8302_ExcoffierLeonard_2085276_Devoir3"
pdflatex -interaction=nonstopmode "$TEX_NAME.tex" > /dev/null
pdflatex -interaction=nonstopmode "$TEX_NAME.tex" > /dev/null
cp "$TEX_NAME.pdf" "../$TEX_NAME.pdf"
echo "PDF written to $TEX_NAME.pdf"
