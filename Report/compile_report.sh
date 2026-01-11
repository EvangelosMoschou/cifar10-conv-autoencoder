#!/bin/bash
# Script για compile του XeLaTeX report

cd "$(dirname "$0")"

echo "Compiling Autoencoder_CIFAR10_Report.tex with XeLaTeX..."
    
# Compile 3 φορές για σωστά references
xelatex -interaction=nonstopmode Autoencoder_CIFAR10_Report.tex
xelatex -interaction=nonstopmode Autoencoder_CIFAR10_Report.tex
xelatex -interaction=nonstopmode Autoencoder_CIFAR10_Report.tex
    
# Καθαρισμός βοηθητικών αρχείων
rm -f *.aux *.log *.out *.toc *.lof *.lot

echo "Done! Check CNN_CIFAR10_Report.pdf"
