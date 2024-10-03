library(tidyverse)
#library(hypervolume)
library(tidysdm)
library(torch)
library(missForest)

n_spec <- 200

squamate_train <- read_csv("output/squamate_training2.csv")
squamate_train <- squamate_train |>
  group_by(Binomial) |>
  mutate(count = n())

table(squamate_train$count)

spec_samp <- squamate_train |>
  distinct(Binomial, .keep_all = TRUE) |>
  filter(count %in% c(4, 800)) |>
  group_by(count) |>
  sample_n(size = n_spec)

squamate_train_reduced <- squamate_train |>
  filter(Binomial %in% spec_samp$Binomial)

test <- squamate_train_reduced |>
  filter(Binomial == squamate_train_reduced$Binomial[1])

bad_vars <- c("CHELSA_swe_1981-2010_V.2.1", "CHELSA_fcf_1981-2010_V.2.1")
env_dat <- test |>
  ungroup() |>
  select(starts_with("CHELSA_"), -all_of(bad_vars)) 

env_dat_mat <- missForest(as.matrix(env_dat))$ximp
env_dat_mat <- scale(env_dat_mat)
nas <- apply(env_dat_mat, 2, function(x) !any(is.finite(x)))
env_dat_mat <- env_dat_mat[ , !nas]

hyp_mod <- hypervolume(env_dat_mat, chunk.size = 10000)
