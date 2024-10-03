source("R/load_and run_models.R")
source("R/geo_utils_and_data.R")
source("R/plotting.R")

library(tidymodels)
library(probably)
library(tidysdm)

library(rayshader)
library(patchwork)

env_vae <- load_nichencoder_vae()
flow_2 <- load_nichencoder_flow_rectified()
geode <- load_geode_flow()

training_data <- load_training_data()
test_data <- load_test_data()
polygons <- load_species_polygons()
metadata <- load_species_metadata()

metadata <- metadata |>
  filter(Area > 2500, Area < 2e+6) |>
  left_join(training_data |>
              group_by(Binomial) |>
              summarise(n2 = n()) |>
              ungroup()) |>
  mutate(fewshot = ifelse(n2 == 4, TRUE, FALSE)) |>
  mutate(area_cat = cut_interval(log(Area + 1), 3),
         lat_cat = cut_interval(abs(mid_lat), 3))

spec_sample <- metadata |>
  group_by(fewshot, area_cat, lat_cat) |>
  slice_sample(n = 25)

spec_dat <- get_species_data(spec_sample$Binomial[400], training_data, test_data, env_vae, polygons)

problem_vars <- c("CHELSA_fcf_1981-2010_V.2.1", "CHELSA_swe_1981-2010_V.2.1")

evaluate_and_plot <- function(species_name, output_dir = "output/model_results", fewshot = FALSE) {
  
  data_file <- file.path(output_dir, paste0(species_name, ".rds"))
  env_plot_file <- file.path(output_dir, paste0(species_name, "_env_plot.png"))
  geo_plot_file <- file.path(output_dir, paste0(species_name, "_geo_plot.png"))
  
  try_it <- try({
  spec_dat <- get_species_data(species_name, training_data, test_data, env_vae, polygons)
  predicts <- make_model_predictions(spec_dat, env_vae, flow_2, geode, problem_vars)
  
  env_plot <- env_compare_plot(predicts$env_data, spec_dat$train, 
                               species = spec_dat$species, problem_vars)
  ragg::agg_png(env_plot_file, width = 1600, height = 1400,
                scaling = 2.3)
  plot(env_plot)
  dev.off()
  
  geo_plot <- geo_compare_plot(predicts$geo_data, spec_dat$species,
                               predicts$metrics)
  ragg::agg_png(geo_plot_file, width = 1600, height = 1000,
                scaling = 2.3)
  if(fewshot) {
    plot(geo_plot$combined_train_test)
  } else {
    plot(geo_plot$combined_test) 
  }
  dev.off()
  })
  
  write_rds(list(spec_dat = spec_dat, predict_dat = predicts,
                 plot_data = list(env_plot = env_plot,
                                  geo_plots = geo_plot)),
            data_file)
  
  invisible(NULL)
}

#evaluate_and_plot(spec_sample$Binomial[400], fewshot = TRUE)

spec_sample <- spec_sample |>
  ungroup() |>
  slice_sample(prop = 1)
walk2(spec_sample$Binomial, spec_sample$fewshot,
      ~ evaluate_and_plot(.x, fewshot = .y),
     .progress = TRUE)


geo_plot <- geo_compare_plot(geo_data, spec_dat$species, metrics)
ragg::agg_png("test.png", width = 1600, height = 1000,
              scaling = 2.3)
plot(geo_plot$combined_train_test)
dev.off()

var_plots <- list()
var_mat <- matrix(colnames(env_samp)[-which(colnames(env_samp) %in% problem_vars)], nrow = 15, ncol = 2, byrow = TRUE)
for(j in 1:nrow(var_mat)) {
  dat <- as.data.frame(env_samp[ , var_mat[j, ], drop = FALSE]) |>
    slice_sample(n = 1000) |>
    mutate(type = "predictions") |>
    bind_rows(as.data.frame(spec_dat$test[ , var_mat[j, ], drop = FALSE]) |>
                mutate(type = "truth")) |>
    slice_sample(prop = 1)
  var_plots[[j]] <- ggplot(dat, aes(.data[[var_mat[j, 1]]], .data[[var_mat[j, 2]]])) +
    geom_point(aes(colour = type), alpha = 0.5) +
    theme_minimal() +
    theme(axis.title = element_text(size = 6))
}

p <- wrap_plots(var_plots, nrow = 4, ncol = 4, guides = "collect") +
  plot_annotation(title = spec_dat$species)
plot(p)

p1 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
  geom_sf(data = geo_data$countries, fill = "grey90", colour = "black") +
  geom_sf(colour = NA) +
  geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
  # geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "red", size = 0.1, shape = 3,
  #         alpha = 0.05) +
  scale_fill_viridis_c(option = "B") +
  coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
           ylim = geo_data$extent[c("ymin", "ymax")]) +
  theme_minimal()
p1

p2 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
  geom_sf(data = geo_data$countries, fill = "grey90", colour = "black") +
  geom_sf(colour = NA) +
  geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
  geom_sf(data = geo_data$truth_sf, inherit.aes = FALSE, colour = "white", size = 0.5, 
          alpha = 0.8) +
  geom_sf(data = geo_data$truth_sf, inherit.aes = FALSE, colour = "green", size = 0.02, 
          alpha = 0.8) +
  scale_fill_viridis_c(option = "B") +
  coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
           ylim = geo_data$extent[c("ymin", "ymax")]) +
  theme_minimal()
p2

p <- p1 + p2 + plot_layout(guides = "collect")
p

p1 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
  geom_sf(colour = NA) +
  geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
  # geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "red", size = 0.1, shape = 3,
  #         alpha = 0.05) +
  scale_fill_viridis_c(option = "B") +
  coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
           ylim = geo_data$extent[c("ymin", "ymax")]) +
  theme_minimal()
p1
gg1 <- plot_gg(p1, preview = TRUE)

plot_gg(p1)
