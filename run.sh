#!/bin/bash

set -e

cd "$(dirname "$0")"

uv run src/main.py

echo "All scripts completed."

echo "Compiling LaTeX..."
cd report
TEX_NAME="MTH8302_ExcoffierLeonard_2085276_Devoir3"
pdflatex -interaction=nonstopmode "$TEX_NAME.tex" > /dev/null
pdflatex -interaction=nonstopmode "$TEX_NAME.tex" > /dev/null
cp "$TEX_NAME.pdf" "../$TEX_NAME.pdf"
echo "PDF written to $TEX_NAME.pdf"