rm -f forest_mscn-figure0*
rm -f *.pdf
pdflatex -shell-escape forest_mscn.tex
xdg-open forest_mscn-figure0.pdf
