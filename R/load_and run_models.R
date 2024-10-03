library(torch)
library(tidyverse)
library(deSolve)
library(cli)
library(tidymodels)
library(probably)
library(sf)

load_nichencoder_flow_rectified <- function(checkpoint_dir = "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_distill_7d", 
                                            device = "cuda") {
  old_opt <- options(torch.serialization_version = 2)
  checkpoint_files <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  checkpoints <- file.info(checkpoint_files)
  most_recent <- which.max(checkpoints$mtime)
  checkpoint <- checkpoint_files[most_recent]
  flow_2 <- torch_load(checkpoint)
  flow_2 <- flow_2$to(device = device)
  options(old_opt)
  flow_2
}

load_geode_flow <- function(checkpoint_dir = "output/checkpoints/geo_env_model_3", device = "cuda") {
  files_geo <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  file_info_geo <- file.info(files_geo)
  latest_geo <- which.max(file_info_geo$mtime)
  mod_file_geo <- files_geo[latest_geo]
  geode <- torch_load(mod_file_geo)
  geode <- geode$to(device = device)
  attr(geode, "scaling") <- read_rds("output/chelsa_geo_scaling.rds")
  geode
}

load_nichencoder_vae <- function(checkpoint_dir = "data/env_vae_trained_fixed2_alpha_0.5_32d.to", 
                                 device = "cuda") {
  env_vae <- torch_load(checkpoint_dir)
  env_vae <- env_vae$to(device = device)
  attr(env_vae, "active_dims") <- read_rds("output/squamate_env_mani2.rds")
  attr(env_vae, "scaling") <- read_rds("output/squamate_env_scaling2.rds")
  attr(env_vae, "species") <- read_csv("output/training_species.csv")
  env_vae
}

run_nichencoder <- function(latent, env_vae = NULL, flow_2 = NULL, n = 10000, device = "cuda", gradient = FALSE, recode = FALSE) {
  
  if(!is.matrix(latent)) {
    stop("latent must be a matrix")
  }
  if(dim(latent)[2] != env_vae$spec_embed_dim) {
    stop("latent must have ", env_vae$spec_embed_dim, " columns")
  }
  
  if(is.null(env_vae)) {
    env_vae <- load_nichencoder_vae(device = device)
  }
  
  if(is.null(flow_2)) {
    flow_2 <- load_geode_flow(device = device)
  }
  
  if(is.null(dim(latent)) || dim(latent)[1] == 1) {
    latent <- matrix(as.vector(latent), nrow = n, ncol = length(latent), byrow = TRUE)
  }
  
  scaling <- attr(env_vae, "scaling")
  active_dims <- attr(env_vae, "active_dims")
  
  if(gradient) {
    latent_tens <- torch_tensor(latent, device = device)
    samp1 <- torch_randn(n, length(active_dims), device = device)
    samp2 <- flow_2$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 2)
    samp <- matrix(0, ncol = env_vae$latent_dim, nrow = n)
    samp[ , active_dims] <- samp2$trajectories[2, , ]
    samp <- torch_tensor(samp, device = device)
    decoded <- env_vae$decoder(z = samp, s = latent_tens)
    if(recode) {
      reencoded <- env_vae$encoder(y = decoded, s = latent_tens)
      resamp <- torch_randn_like(reencoded$logvars) * torch_exp(reencoded$logvars) + reencoded$means
      decoded <- env_vae$decoder(z = resamp, s = latent_tens)
    }
  } else {
    with_no_grad({
      latent_tens <- torch_tensor(latent, device = device)
      samp1 <- torch_randn(n, length(active_dims), device = device)
      samp2 <- flow_2$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 2)
      samp <- matrix(0, ncol = env_vae$latent_dim, nrow = n)
      samp[ , active_dims] <- samp2$trajectories[2, , ]
      samp <- torch_tensor(samp, device = device)
      decoded <- env_vae$decoder(z = samp, s = latent_tens)
      if(recode) {
        reencoded <- env_vae$encoder(y = decoded, s = latent_tens)
        resamp <- torch_randn_like(reencoded$logvars) * torch_exp(reencoded$logvars) + reencoded$means
        decoded <- env_vae$decoder(z = resamp, s = latent_tens)
      }
    })
  }
  env_pred <- as.matrix(decoded$cpu())
  colnames(env_pred) <- names(scaling$means)
  env_pred <- t((t(env_pred) * scaling$sd) + scaling$means)
  return(env_pred)
}

run_geode <- function(geode = NULL, env, steps = 500, problem_vars = NULL, device = "cuda",
                      terminal_only = TRUE) {
  
  if(!is.matrix(env)) {
    stop("env must be a matrix")
  }
  if(dim(env)[2] != 32) {
    stop("env matrix must have 32 columns")
  }
  
  if(is.null(geode)) {
    geode <- load_geode_flow(device = device)
  }
  
  geo_scaling <- attr(geode, "scaling")
  
  if(!is.null(problem_vars)) {
    if(length(problem_vars) > 0) {
      for(i in problem_vars) {
        env[ , i] <- geo_scaling$mean[i][[1]]  
      }
    }
  }
  
  mu <- unlist(geo_scaling$mean)[-1:-2]
  s <- unlist(geo_scaling$sd[-1:-2])
  #geo_sd <- unlist(geo_scaling$sd[1:2])
  with_no_grad({
    n <- dim(env)[1]
    env <- torch_tensor(t((t(env) - mu) / s), device = device)
    samp <- torch_randn(n, 2) * 1.1
    trajs <- geode$sample_trajectory(initial_vals = samp, env_vals = env, steps = steps)
  })
  if(terminal_only) {
    return(
      t((t(trajs$trajectories[steps, , ]) * unlist(geo_scaling$sd)[1:2]) + unlist(geo_scaling$mean)[1:2])
    )
  } else{
    trajs$trajectories <- sweep(trajs$trajectories, 1,
                                function(x) t((t(x) * unlist(geo_scaling$sd)[1:2]) + unlist(geo_scaling$mean)[1:2]),
                                simplify = FALSE)
    return(trajs$trajectories)
  }
  
}

load_training_data <- function(training_data = "output/squamate_training2.csv") {
  read_csv(training_data)
}

load_test_data <- function(test_data = "output/squamate_testing2.csv") {
  read_csv(test_data)
}

load_species_polygons <- function(polygons = "data/final_squamate_sf.rds") {
  read_rds(polygons)
}

load_species_metadata <- function(metadat = "output/species_meta.csv") {
  read_csv(metadat)
}

get_species_data <- function(species_name, training_data = NULL, 
                             test_data = NULL, 
                             env_vae = NULL,
                             polygons = NULL,
                             subsample_train = 0,
                             subsample_test = 0,
                             metadat = NULL) {
  
  if(is.null(polygons)) {
    polygons <- load_species_polygons()
  }
    if(is.null(training_data)) {
    training_data <- load_training_data()  
  } 
  if(is.null(test_data)) {
    test_data <- load_test_data()
  }
  if(is.null(metadat)) {
    metadat <- load_species_metadata()
  }
  
  if(is.numeric(species_name)) {
    n_train <- species_name[1]
    if(has_name(species_name, "test")) {
      species_name <- sample(unique(test_data$Binomial), species_name["test"])
    } else {
      species_name <- sample(unique(training_data$Binomial), species_name)
    }
  }
  
  train <- training_data |>
    filter(Binomial %in% species_name) |>
    rename(species = Binomial)
  
  test <- test_data |>
    filter(Binomial %in% species_name)  |>
    rename(species = Binomial)
  
  polygons <- polygons |>
    filter(Binomial %in% species_name)  |>
    rename(species = Binomial)
  
  metadat <- metadat |>
    filter(Binomial %in% species_name)  |>
    rename(species = Binomial)
  
  if(subsample_train > 0) {
    if(subsample_train > 1) {
      train <- train |>
        group_by(species) |>
        slice_sample(n = subsample_train) |>
        ungroup()
    } else {
      train <- train |>
        group_by(species) |>
        slice_sample(prop = subsample_train) |>
        ungroup()
    }
  }
  
  if(subsample_test > 0) {
    if(subsample_test > 1) {
      test <- test |>
        group_by(species) |>
        slice_sample(n = subsample_test) |>
        ungroup()
    } else {
      test <- test |>
        group_by(species) |>
        slice_sample(prop = subsample_test) |>
        ungroup()
    }
  }
  
  prop_nas_train <- train |>
    summarise(across(starts_with("CHELSA_"), ~sum(is.na(.x))/n()))
  probls_train <- colnames(prop_nas_train)[prop_nas_train > 0.5]
  
  prop_nas_test <- test |>
    summarise(across(starts_with("CHELSA_"), ~sum(is.na(.x))/n()))
  probls_test <- colnames(prop_nas_test)[prop_nas_test > 0.5]
  
  if(is.null(env_vae)) {
    env_vae <- load_nichencoder_vae(device = "cpu")
  }
  
  spec_latents <- as.matrix(env_vae$species_embedder_mean$weight$cpu())
  colnames(spec_latents) <- paste0("L", 1:ncol(spec_latents))
  species_df <- attr(env_vae, "species")
  spec <- species_name
  species_df <- species_df |>
    bind_cols(spec_latents[species_df$id, ] |>
                as.data.frame()) |>
    filter(species %in% spec) |>
    select(-species, -id) |>
    as.matrix()
  
  list(species = species_name, latent = species_df, train = train, test = test,
       polygons = polygons, problem_vars_train = probls_train, 
       problem_vars_test = probls_test,
       metadata = metadat)
}

make_model_predictions <- function(spec_dat, env_vae, flow_2, geode, problem_vars) {
  env_samp <- run_nichencoder(spec_dat$latent, env_vae = env_vae, flow_2 = flow_2)
  geo_samp <- run_geode(geode, env_samp, steps = 250, problem_vars = problem_vars)
  
  train_points <- spec_dat$train |>
    select(X, Y) |>
    st_as_sf(coords = c("X", "Y"), crs = 4326)
  test_points <- spec_dat$test |>
    select(X, Y) |>
    st_as_sf(coords = c("X", "Y"), crs = 4326)
  geo_data <- localize(spec_dat$polygons, geo_samp, train_points = train_points,
                       test_points = test_points)
  
  cli_progress_message("Generating hex summaries...")
  
  hex_data <- geo_data$hex_polys |>
    as_tibble() |>
    select(prop_pred, prop_train, prop_test, count_train, count_test) |>
    replace_na(list(prop_pred = 0, prop_train = 0, prop_test = 0, count_train = 0, count_test = 0)) |>
    mutate(pres_train = factor(ifelse(count_train > 0, "Yes", "No"), levels = c("Yes", "No")),
           pres_test = factor(ifelse(count_test > 0, "Yes", "No"), levels = c("Yes", "No")))
  
  cli_progress_message("Finding calibration threshold...")
  thres <- threshold_perf(hex_data,
                          pres_test,
                          prop_pred,
                          thresholds = seq(0, 1, length.out = 2000))
  thres_choose <- thres |>
    filter(.metric == "j_index") |>
    slice_max(.estimate, n = 1)
  
  hex_data <- hex_data |>
    mutate(pres_pred = make_two_class_pred(prop_pred, c("Yes", "No"),
                                           threshold = thres_choose$.threshold[[1]]),
           weight_test = frequency_weights(count_test + 1),
           weight_train = frequency_weights(count_train + 1))
  
  cli_progress_message("Calculating metrics...")
  metrics <- bind_rows(
    j_index(hex_data, pres_test, pres_pred, case_weights = weight_test),
    accuracy(hex_data, pres_test, pres_pred, case_weights = weight_test),
    roc_auc(hex_data, pres_test, prop_pred, case_weights = weight_test),
    f_meas(hex_data, pres_test, pres_pred, case_weights = weight_test),
    kap(hex_data, pres_test, pres_pred, case_weights = weight_test)
  )
  
  list(env_data = env_samp, geo_data = geo_data, hex_data = hex_data, metrics = metrics)
}


