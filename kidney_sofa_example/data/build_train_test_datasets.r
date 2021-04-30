library(tidyverse)
library(rlang)
library(magrittr)
library(corrplot)  # A package that allows to make nice looking correlation plots.
library(ggplot2)

dataFile <- 
  "dataset_final-04052020.rds"

target <- "sofa_kidney"

frac_test <- .2


data <- 
  readRDS(dataFile) %>%
  ungroup() %>%
  #rename targeted sofa score to target
  rename(target = !! sym(target)) %>%
  #remove other sofa scores
  select(-starts_with("sofa")) %>%
  arrange(id, charttime) %>%
  select(-charttime) %>%
  #remove duplicates
  distinct() %>%
  mutate(test_sample = sort(rep(c(FALSE, TRUE), 
                           times = round(n() * c((1 - frac_test), frac_test), 0)))) %>%
  group_by(id) %>%
  mutate(test_sample = all(test_sample)) %>%
  ungroup()


#check distributions
pdf("check_train-test_similarity.pdf")

ggplot(data = data %>%
              group_by(target, test_sample) %>%
              summarize(freq = n()) %>%
              ungroup() %>%
              group_by(test_sample) %>%
              mutate(prop = freq / sum(freq) * 100) %>%
              ungroup()) +
  theme_bw() +
  theme(legend.position = "top") +
  geom_col(aes(x = target, y= prop,  fill=test_sample), position = "dodge")

for (X in names(data)[2:45]) {

  plot <- 
    ggplot(data = data) +
    theme_bw() +
    theme(legend.position = "top") + 
    geom_density(aes(x = !! sym(X), col = test_sample))
  
  print(plot)
  
}

z <- lapply(split(data[2:46], data$test_sample), cor)

corrplot::corrplot(z[[1]], method = "ellipse", type = "upper", title = "train")
corrplot::corrplot(z[[2]], method = "ellipse", type = "upper", title = "test")

dev.off()

#save unstandardized test data (for plots etc.) 

data%>%
  filter(test_sample) %>%
  select(-test_sample, -id) %T>%
  saveRDS(., "dataset-test_nostd.rds")

data%>%
  filter(!test_sample) %>%
  select(-test_sample, -id) %T>%
  saveRDS(., "dataset-train_nostd.rds")

#standardize and save data sets
data %<>%
  mutate(across(.cols = names(data)[2:45], .fns = ~ as.vector(scale(.x)), .names ="{col}"))
  
data %>%
  filter(!test_sample) %>%
  select(-test_sample, -id) %T>%
  saveRDS(.,"dataset-train_std.rds")

data %>%
  filter(test_sample) %>%
  select(-test_sample, -id) %T>%
  saveRDS(., "dataset-test_std.rds")

#remove circular features
data %>%
  filter(!test_sample) %>%
  select(-test_sample, -id, -bili) %T>%
  saveRDS(.,"dataset-train_std_nocirc.rds")

data %>%
  filter(test_sample) %>%
  select(-test_sample, -id, -bili) %T>%
  saveRDS(., "dataset-test_std_nocirc.rds")

##build ablation set
data %>%
  filter(test_sample) %>%
  select(-test_sample, -id) %T>%
  mutate(crea = 0,
         urine24 = 0) %>%
  saveRDS(., "dataset-ablation_zeroCreaUrine24_std.rds")