#!/bin/bash
# Script για compile του XeLaTeX report

cd "$(dirname "$0")"

echo "Compiling CNN_CIFAR10_Report.tex with XeLaTeX..."

# Compile 3 φορές για σωστά references
xelatex -interaction=nonstopmode CNN_CIFAR10_Report.tex
xelatex -interaction=nonstopmode CNN_CIFAR10_Report.tex
xelatex -interaction=nonstopmode CNN_CIFAR10_Report.tex

# Καθαρισμός βοηθητικών αρχείων
rm -f *.aux *.log *.out *.toc *.lof *.lot

echo "Done! Check CNN_CIFAR10_Report.pdf"
