This directory contains all files need to rerun the analyses related to the 
kidney (renal) sofa example presented in the book.

The most important files are the two R Markdown files (we recommend using RStudio
to open them):
[1] kidney_circularity.Rmd
[2] kidney_distillation.Rmd

which store the necessary R code to replicate the results presented
in:

[1] Chapter 2.4.3 (Circularity in Data Annotation Prediction)
[2] Chapter 2.4.3 (Circularity in Machine Learning Prediction)


The directory data/ contains the raw data (train and test) needed to train the
DNN defined in /train-pytorch. In order to train this model from scratch you 
need to run train-pytorch/train_network.py. This scripts trains the DNN,
generates predictions (for test and train data) and also conducts the ablation 
study experiments. The results are stored in models/ and predictions/.
The script prepare_analysis_dataset.Rmd combines these results into the three 
analysis data sets train_set_with_sofa_predictions.rds, 
test_set_with_sofa_predictions.rds and ablation_set_with_sofa_predictions.rds 
which are used for analysis.

REMARK: If you knit the Rmd files you will obtain a pdf with the same name as
the script along with the figures in png,svg and pdf stored in figures/. The
computationally more expansive sections are cached in *_cache/ directories.