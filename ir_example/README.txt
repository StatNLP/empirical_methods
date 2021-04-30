This directory contains all files need to rerun the analyses related to the 
information retrieval example presented in the book.

The most important files are the two R Markdown files (we recommend using RStudio
to open them):
[1] ir_circularity.Rmd
[2] ir_distill.Rmd

which store the necessary R code to replicate the results presented
in:

[1] Chapter 2.4.3 (Circularity in Data Annotation Prediction)
[2] Chapter 2.4.3 (Circularity in Machine Learning Prediction)


The directory data/ containes the ready for analysis data sets otained 
from the experiments.

REMARK: If you knit the Rmd files you will obtain a pdf with the same name as
the script along with the figures in png,svg and pdf stored in figures/. The
computationally more expensive sections are cached in *_cache/ directories.