This directory contains all files need to rerun the analyses related to Kreutzer
 et al data presented in the book.

The most important files are the three R Markdown files
(we recommend using RStudio to open them):

[1] kreutzer_reliability_data_annotaton.Rmd
[2] kreutzer_reliability_model.Rmd
[3] kreutzer_significance.Rmd

which store the necessary R code to replicate the results presented
in:

[1] Chapter 3.3 (Reliability of Data Annotation Performance)
[2] Chapter 3.3 (Reliability of Model Prediction Performance)
[3] Chapter 4.3 (Likelihood Ratio Tests using LMEMs).

The directory data/ contains the ready for analysis data sets obtained
from the experiments.

REMARK: If you knit the Rmd files you will obtain a pdf with the same name as
the script along with the figures in png,svg and pdf stored in figures/. The
computationally more expansive sections are cached in *_cache/ directories.