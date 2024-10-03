library(tidyverse)
library(tidymodels)
library(xgboost)
library(ranger)

source("R/load_and run_models.R")

#metadata <- load_species_metadata()
training_data <- load_training_data()

spec_data <- training_data |>       
  group_by(Binomial) |>
  summarise(n2 = n()) |>
  ungroup() |>
  filter(n2 > 4)

env_vae <- load_nichencoder_vae()

spec_data <- get_species_data(training_data$Binomial, training_data, env_vae = env_vae)

scaling <- attr(env_vae, "scaling")
species_df <- attr(env_vae, "species")

species_name <- spec_data$train$species[1]
l_df <- species_df |>
  bind_cols(spec_data$latent)
calculate_summaries <- function(species_name) {
  
  mat <- spec_data$train |>
    filter(species == species_name) |>
    select(starts_with("CHELSA")) |>
    as.matrix()
  mat <- scale(mat, center = scaling$means, scale = scaling$sd)
  mat[is.na(mat)] <- 0
  
  cv <- cov(mat)
  means <- colMeans(mat)
  
  matrix(c(as.vector(cv), means), nrow = 1) |>
    as.data.frame()
  
}

summaries <- map(spec_data$train$species[1:10], calculate_summaries, .progress = TRUE)

summaries <- map(species_df$species, calculate_summaries, .progress = TRUE) |>
  list_rbind()
summaries <- l_df |>
  bind_cols(summaries)

write_csv(summaries, "output/env_summaries.csv")

set.seed(123)
split <- initial_split(summaries, prop = 0.8)
train <- training(split)
test <- testing(split)

dtrain <- xgb.DMatrix(data = train |> select(starts_with("V")) |> as.matrix(),
                      label = train |> select(starts_with("L")) |> as.matrix())

dtest <- xgb.DMatrix(data = test |> select(starts_with("V")) |> as.matrix(),
                     label = test |> select(starts_with("L")) |> as.matrix())

mod <- xgb.train(data = dtrain,
               evals = list(train = dtrain, test = dtest),
               verbose = TRUE,
               nrounds = 500,
               early_stopping_rounds = 5)



xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(),
  sample_size = tune(),
  mtry = tune(),  
  learn_rate = tune()                          ## step size
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  sample_size = sample_prop(),
  finalize(mtry(), train),
  learn_rate(),
  size = 30
)

xgb_grid

xgb_spec

watchlist <- list(train = summaries |> select(starts_with("V")) |> as.matrix(), 
                  test = summaries |> select(starts_with("L")) |> as.matrix())

mod <- xgboost(data = summaries |> select(starts_with("V")) |> as.matrix(),
               label = summaries |> select(starts_with("L")) |> as.matrix(),
               verbose = TRUE,
               nrounds = 200)
