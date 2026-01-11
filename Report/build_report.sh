#!/usr/bin/env bash
# build_report.sh - compile Knn, NC Report.tex and clean auxiliary files
# Usage: ./build_report.sh [--remove-png]
# If --remove-png is passed the script will delete AUTH_logo.png after building.

set -euo pipefail
cd "$(dirname "$0")"

TEXFILE="Autoencoder_CIFAR10_Report.tex"
OUTPDF="Autoencoder_CIFAR10_Report.pdf"

echo "Compiling $TEXFILE..."
# Run xelatex twice to resolve references; suppress terminal chatter
xelatex -interaction=nonstopmode -halt-on-error "$TEXFILE" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error "$TEXFILE" >/dev/null

echo "Compilation finished. Cleaning auxiliary files..."
# Remove common auxiliary files
rm -f "Autoencoder_CIFAR10_Report.aux" "Autoencoder_CIFAR10_Report.log" "Autoencoder_CIFAR10_Report.out" "Autoencoder_CIFAR10_Report.toc"

if [ "${1-}" = "--remove-png" ]; then
  echo "Removing AUTH_logo.png as requested..."
  rm -f AUTH_logo.png
fi

echo "Done. Generated: $OUTPDF"
