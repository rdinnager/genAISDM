library(tidyverse)
library(tidymodels)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(536567678)

squamates <- read_rds("output/squamate_final_data_w_env_samps2.rds")

squamates <- squamates |>
  select(Binomial, env_samps) |>
  unnest(env_samps) |>
  select(-contains("kg"), -contains("lgd"),
         #-contains("gdgfgd"), -contains("gddlgd"),
         -contains("fgd")) |>
  group_by(Binomial) |>
  slice_sample(n = 1000) |>
  mutate(n = n()) |>
  ungroup() |>
  filter(n > 100)

squamate_split_spec <- group_initial_split(squamates, Binomial, prop = 0.9)
squamate_train_spec <- training(squamate_split_spec)
squamate_test_spec <- testing(squamate_split_spec)

squamate_split <- initial_split(squamate_train_spec, prop = 0.8,
                                strata = Binomial, pool = 0)

squamate_train <- training(squamate_split)
squamate_test <- testing(squamate_split) |>
  mutate(type = "within_species") |>
  bind_rows(squamate_test_spec |>
              mutate(type = "between_species"))

### Add some rare species
squamate_split_spec2 <- group_initial_split(squamate_train, Binomial, prop = 0.9)
squamate_train <- training(squamate_split_spec2)
squamate_test_spec2 <- testing(squamate_split_spec2)
squamate_split2 <- initial_split(squamate_test_spec2, prop = 0.005,
                                 strata = Binomial, pool = 0)
squamate_extra_train <- training(squamate_split2)

squamate_train <- bind_rows(squamate_train, squamate_extra_train)
squamate_test <- bind_rows(squamate_test, testing(squamate_split2))

write_csv(squamate_train, "output/squamate_training2.csv")
write_csv(squamate_test, "output/squamate_testing2.csv")
