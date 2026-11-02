#!/bin/bash
set -e
P=$HOME/ptx-deconfound/paper_ieee; O=$HOME/work/out; rm -rf $O; mkdir -p $O/src
cd $P && ./build.sh >/dev/null && python3 $HOME/work/highlight.py main_v0_submitted.pdf main.pdf $O/Highlighted_PDF_Access-2026-49559.pdf 2>/dev/null | tail -1
cp main.pdf $O/Main_Manuscript_Access-2026-49559.pdf
cp main.tex refs.bib ieeeaccess.cls IEEEtran.cls IEEEtranM.bst logo.png notaglinelogo.png bullet.png salman.png khalaf.png razooqi.png abdulateef.png main.bbl $O/src/
mkdir -p $O/src/figures && for f in $(grep -o 'figures/[A-Za-z0-9_]*\.pdf' main.tex | sort -u); do cp $f $O/src/figures/; done
(cd $O/src && zip -qr ../Main_Manuscript_LaTeX_Access-2026-49559.zip .)
rm -rf $O/src
cd $HOME/work && python3 build_response.py >/dev/null && cp Response_to_Reviewers_Access-2026-49559.docx $O/
mkdir -p $O/new_figures_png && cp $HOME/ptx-deconfound/paper/figures/fig_posthoc_*.png $O/new_figures_png/
ls -la $O $O/new_figures_png
