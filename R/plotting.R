library(tidyverse)
library(patchwork)
library(gt)

env_compare_plot <- function(env_samp, env_test, species, problem_vars) {
  var_plots <- list()
  var_mat <- matrix(colnames(env_samp)[-which(colnames(env_samp) %in% problem_vars)], nrow = 15, ncol = 2, byrow = TRUE)
  var_mat[ , 2] <- sample(var_mat[ , 2])
  var_mat[ , 1] <- sample(var_mat[ , 1])
  for(j in 1:nrow(var_mat)) {
    dat <- as.data.frame(env_samp[ , var_mat[j, ], drop = FALSE]) |>
      slice_sample(n = 1000) |>
      mutate(type = "predictions") |>
      bind_rows(as.data.frame(env_test[ , var_mat[j, ], drop = FALSE]) |>
                  mutate(type = "training data")) 
    var_plots[[j]] <- ggplot(dat, aes(.data[[var_mat[j, 1]]], .data[[var_mat[j, 2]]])) +
      geom_point(aes(colour = type, alpha = type)) +
      scale_alpha_manual(values = c(predictions = 0.25, "training data" = 0.9)) +
      theme_minimal() +
      theme(axis.title = element_text(size = 6))
  }
  var_plots[[16]] <- guide_area()
  
  p <- wrap_plots(var_plots, nrow = 4, ncol = 4, guides = "collect") +
    plot_annotation(title = species)
  p
}

geo_compare_plot <- function(geo_data, species = "Example Species", metrics = NULL) {
  
  world_inset <- ggplot(s2::s2_data_countries() |>
                          st_as_sf() |>
                          st_transform("+proj=sinu +R=6371007.181 +wktext")) +
    geom_sf(fill = "grey20", colour = NA) +
    geom_sf(data = geo_data$ecoregions, fill = "hotpink", colour = "hotpink") +
    theme(plot.background = element_rect(fill = "white"))
  
  p1 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
    geom_sf(data = geo_data$countries, fill = "grey90", colour = "black") +
    geom_sf(colour = NA) +
    geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
    # geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "red", size = 0.1, shape = 3,
    #         alpha = 0.05) +
    scale_fill_viridis_c(name = "Predicted Occurrence\nPoint Proportion", option = "B") +
    coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
             ylim = geo_data$extent[c("ymin", "ymax")]) +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white", colour = NA)) +
    ggtitle("NicheFlow Predictions")

  p2 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
    geom_sf(data = geo_data$countries, fill = "grey90", colour = "black") +
    geom_sf(colour = NA) +
    geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
    geom_sf(data = geo_data$test_points, inherit.aes = FALSE, colour = "white", size = 0.4, 
            alpha = 0.8) +
    geom_sf(data = geo_data$test_points, inherit.aes = FALSE, colour = "green", size = 0.1, 
            alpha = 0.8) +
    scale_fill_viridis_c(name = "Predicted Occurrence\nPoint Proportion", option = "B") +
    coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
             ylim = geo_data$extent[c("ymin", "ymax")]) +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white", colour = NA)) +
    ggtitle("Test Occurrences")
  
  p3 <- ggplot(geo_data$ecoregions |> mutate(prop_pred = 0), aes(fill = prop_pred)) +
    geom_sf(data = geo_data$countries, fill = "grey90", colour = "black") +
    geom_sf(colour = NA) +
    geom_sf(data = geo_data$hex_polys |> drop_na(prop_pred), colour = NA) +
    geom_sf(data = geo_data$train_points, inherit.aes = FALSE, colour = "white", size = 2, 
            alpha = 0.6) +
    geom_sf(data = geo_data$train_points, inherit.aes = FALSE, colour = "yellow", size = 1.5, 
            alpha = 0.6) +
    scale_fill_viridis_c(name = "Predicted Occurrence\nPoint Proportion", option = "B") +
    coord_sf(xlim = geo_data$extent[c("xmin", "xmax")],
             ylim = geo_data$extent[c("ymin", "ymax")]) +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white", colour = NA)) +
    ggtitle("Training Occurrences")
  
  design <- 
  "AAABBB
   CCDDEE"
  
  p_test <- p1 + p2 + 
    wrap_table(metrics |> select(Metric = .metric, Estimate = .estimate)) +
    guide_area() + world_inset + plot_layout(design = design,
                                             guides = 'collect',
                                             heights = c(2, 1)) +
    plot_annotation(title = species) 
  
  p_train <- p1 + p3 + 
    wrap_table(metrics |> select(Metric = .metric, Estimate = .estimate)) +
    guide_area() + world_inset + plot_layout(design = design,
                                             guides = 'collect',
                                             heights = c(2, 1)) +
    plot_annotation(title = species)
  
  p_train_test <- p3 + p2 + 
    wrap_table(metrics |> select(Metric = .metric, Estimate = .estimate)) +
    guide_area() + world_inset + plot_layout(design = design,
                                             guides = 'collect',
                                             heights = c(2, 1)) +
    plot_annotation(title = species)
    
  
  # p_test <- ((p1 + p2) & 
  #              theme(legend.position = "right",
  #                    legend.justification = "bottom")) + 
  #   inset_element(world_inset, left = 0.6, right = 0.99,
  #                 bottom = 0.5, top = 0.9,
  #                 align_to = "full") +
  #   plot_layout(guides = "collect") +
  #   plot_annotation(title = species) 
  # p_train <- ((p1 + p3) & theme(legend.position = "right",
  #                              legend.justification = "bottom")) + 
  #   inset_element(world_inset, left = 0.6, right = 0.99,
  #                 bottom = 0.5, top = 0.9,
  #                 align_to = "full") +
  #   plot_layout(guides = "collect") +
  #   plot_annotation(title = species) 
  # p_all <- ((p1 + p3 + p2) & theme(legend.position = "right",
  #                            legend.justification = "bottom")) + 
  #   inset_element(world_inset, left = 0.6, right = 0.99,
  #                 bottom = 0.5, top = 0.9,
  #                 align_to = "full") +
  #   plot_layout(guides = "collect") +
  #   plot_annotation(title = species)
  
  list(pred_only = p1, pred_and_test = p2, pred_and_train = p3, 
       combined_test = p_test, combined_train = p_train,
       combined_train_test = p_train_test)
}