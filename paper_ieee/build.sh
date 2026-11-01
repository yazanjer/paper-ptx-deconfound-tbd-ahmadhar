#!/bin/bash
cd "$(dirname "$0")"; f=${1:-main}
pdflatex -interaction=nonstopmode $f >/dev/null; bibtex $f >/dev/null; pdflatex -interaction=nonstopmode $f >/dev/null; pdflatex -interaction=nonstopmode $f >/dev/null
grep -E '^!' $f.log | sort | uniq -c; grep -E 'Output written' $f.log; grep -c 'undefined' $f.log
