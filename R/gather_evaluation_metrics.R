library(tidyverse)
library(tidymodels)
library(sf)
library(terra)

source("R/load_and run_models.R")

metadata <- load_species_metadata()
training_data <- load_training_data()

metadata <- metadata |>
  filter(Area > 2500, Area < 2e+6) |>
  left_join(training_data |>
              group_by(Binomial) |>
              summarise(n2 = n()) |>
              ungroup()) |>
  mutate(fewshot = ifelse(n2 == 4, TRUE, FALSE)) |>
  mutate(area_cat = cut_interval(log(Area + 1), 3),
         lat_cat = cut_interval(abs(mid_lat), 3))

res_dir <- "output/model_results/"

res_files <- list.files(res_dir, full.names = TRUE, pattern = ".rds")
res_file <- res_files[1]
extract_metrics <- function(res_file) {
  dat <- read_rds(res_file)
  metrics <- dat$predict_dat$metrics |>
    mutate(species = dat$spec_dat$species)
  metrics
}

metrics_df <- map(res_files,
                  extract_metrics,
                  .progress = TRUE) |>
  list_rbind()

metrics_df <- metrics_df |>
  left_join(metadata, by = c("species" = "Binomial"))

write_rds(metrics_df, "output/metrics.rds")

#metrics_df <- read_csv("output/metrics.csv")

plot_df <- metrics_df |>
  filter(.metric %in% c("roc_auc", "j_index", "f_meas")) |>
  mutate(Metric = case_when(.metric == "roc_auc" ~ "AUC",
                            .metric == "j_index" ~ "TSS",
                            .metric == "f_meas" ~ "F-score"),
         Estimate = .estimate,
         `Geographic Range Size` = factor(c("Small", "Medium", "Large")[as.numeric(area_cat)],
                                          levels = c("Small", "Medium", "Large")),
         `Median Latitude` = factor(c("Equatorial", "Middle Latitude", "High Latitude")[as.numeric(lat_cat)],
                                    levels = c("Equatorial", "Middle Latitude", "High Latitude")),
         fewshot_status = ifelse(fewshot, "Data Deficient (Few-shot)", "Data Abundant"))
ragg::agg_png("figures/model_eval.png", height = 900, width = 1400,
              scaling = 2.3)
ggplot(plot_df, 
       aes(Metric, Estimate)) +
  geom_boxplot(aes(fill = `Geographic Range Size`)) +
  geom_point(aes(fill = `Geographic Range Size`), position = position_jitterdodge(jitter.width = 0.2),
             alpha = 0.5,
             shape = 21) +
  facet_grid(rows = vars(`Median Latitude`), cols = vars(fewshot_status)) +
  theme_minimal()
dev.off()

eval_summ <- plot_df |>
  group_by(fewshot_status, Metric, `Geographic Range Size`, `Median Latitude`) |>
  summarise(mean = mean(Estimate), se = sd(Estimate) / sqrt(n()))

eval_data_summ <- plot_df |>
  group_by(fewshot_status, Metric) |>
  summarise(mean = mean(Estimate), se = sd(Estimate) / sqrt(n()))



############ Run ENMTools ###########
library(ENMTools)

res_dir <- "output/model_results/"

res_files <- list.files(res_dir, full.names = TRUE, pattern = ".rds")
res_file <- res_files[2]

chelsa_files <- list.files("data/env/CHELSA-BIOMCLIM+/1981-2010/bio", 
                           full.names = TRUE)
env <- rast(chelsa_files)
problem_vars <- c("CHELSA_fcf_1981-2010_V.2.1", "CHELSA_swe_1981-2010_V.2.1")
env <- env[[!names(env) %in% problem_vars]]

run_enmtools <- function(res_file) {
  dat <- read_rds(res_file)
  bg <- dat$predict_dat$geo_data$ecoregions
  pres <- vect(dat$predict_dat$geo_data$train_points)

  names(env) <- make.names(names(env))
  env_spec <- crop(env, bg |> st_transform(4326))
  env_spec <- scale(env_spec)
  
  spec_ob <- enmtools.species(species.name = dat$spec_dat$species,
                              presence.points = pres,
                              background.points = vect(st_sample(bg, 10000)))

  mod <- enmtools.rf.ranger(spec_ob, env, verbose = TRUE)
    
}
