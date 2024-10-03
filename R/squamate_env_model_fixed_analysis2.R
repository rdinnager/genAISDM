library(tidyverse)
library(torch)
library(ape)
library(phyf)
library(ggmulti)
library(deSolve)
library(uwot)
library(sf)
library(rnaturalearth)
library(patchwork)
library(tidymodels)
library(h3)
library(eks)

world <- ne_countries(scale = 10)
maps <- read_rds("data/final_squamate_sf.rds")
ecoregions <- read_rds("data/maps/ecoregions_valid.rds")

# data(countries110)
# countries110 <- st_make_valid(countries110)

options(torch.serialization_version = 2)
checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_distill_7d"
checkpoint_files <- list.files(checkpoint_fold, full.names = TRUE, pattern = ".pt")
checkpoints <- file.info(checkpoint_files)
most_recent <- which.max(checkpoints$mtime)
checkpoint <- checkpoint_files[most_recent]
flow_2 <- torch_load(checkpoint)
flow_2 <- flow_2$cuda()

checkpoint_dir_geo <- "output/checkpoints/geo_env_model_3"
files_geo <- list.files(checkpoint_dir_geo, full.names = TRUE, pattern = ".pt")
file_info_geo <- file.info(files_geo)
latest_geo <- which.max(file_info_geo$mtime)
mod_file_geo <- files_geo[latest_geo]

env_vae <- torch_load("data/env_vae_trained_fixed2_alpha_0.5_32d.to")
env_vae <- env_vae$cuda()
# flow_1 <- torch_load(mod_file)
# flow_1 <- flow_1$cuda()

options(torch.serialization_version = 3)
geode <- torch_load(mod_file_geo)
geode <- geode$cuda()

active_dims <- read_rds("output/squamate_env_mani2.rds")

spec_latents <- as.matrix(env_vae$species_embedder_mean$weight$cpu())

squamate_train <- read_csv("output/squamate_training2.csv")
species_df <- tibble(id = as.integer(as.numeric(as.factor(squamate_train$Binomial))),
                     species = squamate_train$Binomial) |>
  distinct()

write_csv(species_df, "output/training_species.csv")

scaling <- read_rds("output/squamate_env_scaling2.rds")
geo_scaling <- read_rds("output/chelsa_geo_scaling.rds")

latent_df <- as.data.frame(spec_latents[species_df$id, ])
colnames(latent_df) <- paste0("L", 1:ncol(latent_df))
species_latent_df <- bind_cols(species_df, latent_df)

# median_lats <- squamate_train |>
#   group_by(Binomial) |>
#   summarise(mid_lat = median(Y))
# 
# write_csv(median_lats, "output/squamate_latitudes2.csv")

#tree <- read.tree("data/phylogenies/squamates/squamates.tre")

predict_niche <- function(env_vae, flow_2, n = 10000, latent, scaling, active_dims, device = "cuda") {
  with_no_grad({
    if(is.null(dim(latent)) || dim(latent)[1] == 1) {
      latent <- matrix(as.vector(latent), nrow = n, ncol = length(latent), byrow = TRUE)
    }
    latent_tens <- torch_tensor(latent, device = device)
    samp1 <- torch_randn(n, length(active_dims), device = device)
    samp2 <- flow_2$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 2)
    samp <- matrix(0, ncol = env_vae$latent_dim, nrow = n)
    samp[ , active_dims] <- samp2$trajectories[2, , ]
    samp <- torch_tensor(samp, device = device)
    decoded <- env_vae$decoder(z = samp, s = latent_tens)
    reencoded <- env_vae$encoder(y = decoded, s = latent_tens)
    resamp <- torch_randn_like(reencoded$logvars) * torch_exp(reencoded$logvars) + reencoded$means
    redecoded <- env_vae$decoder(z = resamp, s = latent_tens)
  })
  env_pred <- as.matrix(decoded$cpu())
  colnames(env_pred) <- names(scaling$means)
  env_pred <- t((t(env_pred) * scaling$sd) + scaling$means)
  env_pred
}

problematic <- c(12, 17, 20, 25, 32)

predict_geo <- function(geode, env_pred, geo_scaling, device = "cuda") {
  mu <- unlist(geo_scaling$mean)[-1:-2]
  s <- unlist(geo_scaling$sd[-1:-2])
  #geo_sd <- unlist(geo_scaling$sd[1:2])
  with_no_grad({
    n <- dim(env_pred)[1]
    env <- torch_tensor(t((t(env_pred) - mu) / s), device = device)
    samp <- torch_randn(n, 2) * 1.1
    trajs <- geode$sample_trajectory(initial_vals = samp, env_vals = env, steps = 500)
  })
  trajs
}

spec <- sample(species_df$species, 1)
generate_range_sample <- function(species, n = 10000, geode, env_vae, 
                                  flow_2, scaling, geo_scaling, active_dims,
                                  species_latent_df, squamate_train) {
  
  latent <- species_latent_df |>
    filter(species == spec)
  
  spec_train <- squamate_train |>
    filter(Binomial == spec) |>
    select(-Binomial, -X, -Y, -n)
  spec_xy <- squamate_train |>
    filter(Binomial == spec) |>
    select(X, Y)
  #|>
  #  replace_na(as.list(geo_scaling$mean))
  
  # srec_train_mat <- |>
  #   as.matrix()
  # spec_train[is.na(spec_train)] <- 0
  # 
  prop_nas <- spec_train |>
    summarise(across(everything(), ~sum(is.na(.x))/n()))
  probls <- colnames(prop_nas)[prop_nas > 0.5]
  
  latent_vc <- latent |> select(starts_with("L")) |> unlist()
  pred_niche_vc <- predict_niche(env_vae, flow_2, latent = latent_vc, 
                                 scaling = scaling, active_dims = active_dims)
  
  if(length(probls) > 0) {
    for(i in probls) {
      pred_niche_vc[ , i] <- geo_scaling$mean[i][[1]]  
    }
    for(i in probls) {
      spec_train[ , i] <- geo_scaling$mean[i][[1]]  
    }
  }
  
  trajs <- predict_geo(geode, pred_niche_vc, geo_scaling)
  
  mu_xy <- unlist(geo_scaling$mean[1:2])
  s_xy <- unlist(geo_scaling$sd[1:2])
  coord_pred <- t((t(trajs$trajectories[500, , ]) * (s_xy)) + mu_xy)
  
  list(predictions = coord_pred, truth = spec_xy, env_pred = pred_niche_vc, env_truth = spec_train)
}

coord_predict <- generate_range_sample (spec, n = 10000, geode, env_vae, 
                                     flow_2, scaling, geo_scaling, active_dims,
                                     species_latent_df, squamate_train)

find_equal_area_projection <- function(sf_object) {
  # Ensure the dataset is in WGS84 geographic coordinates
  sf_object <- st_transform(sf_object, 4326)
  
  # Calculate the bounding box and centroid
  bbox <- st_bbox(sf_object)
  centroid <- st_coordinates(st_centroid(st_union(sf_object)))
  lon0 <- centroid[1]
  lat0 <- centroid[2]
  
  # Calculate extent in degrees
  delta_lon <- bbox$xmax - bbox$xmin
  delta_lat <- bbox$ymax - bbox$ymin
  
  # Determine if the dataset has a global extent
  is_global_extent <- delta_lon >= 180 || delta_lat >= 90
  
  if (is_global_extent) {
    # Use Equal Earth projection for global datasets
    proj_str <- "+proj=eqearth +units=m +ellps=WGS84"
  } else if (delta_lat > delta_lon) {
    # Predominantly north-south extent
    # Use Lambert Azimuthal Equal-Area projection
    proj_str <- sprintf(
      "+proj=laea +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      lat0, lon0
    )
  } else {
    # Predominantly east-west extent
    # Use Albers Equal-Area Conic projection
    # Set standard parallels based on dataset's latitude
    std_parallel_1 <- lat0 - delta_lat / 6
    std_parallel_2 <- lat0 + delta_lat / 6
    proj_str <- sprintf(
      "+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      std_parallel_1, std_parallel_2, lat0, lon0
    )
  }
  
  # Reproject the data
  #sf_projected <- st_transform(sf_object, crs = proj_str)
  
  return(proj_str)
}

#ecoreg_border <- st_union(ecoregions)

localize <- function(coord_predict, spec, ecoregions, world, use_preds = FALSE, res = 5) {

  pred_sf <- coord_predict$predictions |>
    as.data.frame() |>
    st_as_sf(coords = c("V1", "V2"), crs = 4326)
  truth_sf <- coord_predict$truth |>
    st_as_sf(coords = c("X", "Y"), crs = 4326)
  if(use_preds) {
    samp_coords <- rbind(pred_sf,
                         truth_sf) |>
      mutate(point = 1)  
  } else {
    samp_coords <- truth_sf |>
      mutate(point = 1)
  }
  
  hex_res <- map(1:10, ~ unique(geo_to_h3(truth_sf, .x)), .progress = TRUE)
  n_hexes <- map_int(hex_res, length)
  res_choice <- which.min(abs(300 - n_hexes))
  hexes <- hex_res[[res_choice]]
  hex_sf <- h3_to_geo_boundary_sf(hexes)
  
  ecoreg <- ecoregions |>
    st_join(hex_sf) |>
    filter(!is.na(h3_index)) |>
    distinct(ECO_NAME, .keep_all = TRUE)
  # ecoreg <- ecoreg |>
  #   st_cast("POLYGON") |>
  #   select(-h3_index) |>
  #   mutate(poly_id = 1:n())
  # ecoreg <- ecoreg |>
  #   st_join(hex_sf) |>
  #   filter(!is.na(h3_index)) |>
  #   distinct(poly_id, .keep_all = TRUE)
  
  hex_points <- pred_sf |>
    mutate(point = 1) |>
    st_join(hex_sf) |>
    group_by(h3_index)
  
  hex_counts_pred <- hex_points |>
    summarise(count = sum(point)) |>
    ungroup() |>
    mutate(prop = count / max(count)) |>
    filter(prop > 0.01) 
  
  # pred_sf_red <- hex_points |>
  #   mutate(count = sum(point)) |>
  #   ungroup() |>
  #   mutate(prop = count / max(count)) |>
  #   filter(prop > 0.01) 
  # test <- st_kcde(pred_sf,  verbose = TRUE)
  # kde_contour <- st_get_contour(test)
  
  hex_polys <- hex_sf |>
    left_join(hex_counts_pred |> as_tibble() |> select(-geometry))
  
  hex_polys <- hex_polys |>
    st_intersection(ecoreg)
  
  proj <- find_equal_area_projection(ecoreg)
  
  ecoreg <- ecoreg |>
    st_transform(proj)
  countries <- world |>
    st_transform(proj)
  hex_polys <- hex_polys |>
    st_transform(proj)
  
  extent <- st_bbox(ecoreg)
  hex_polys <- hex_polys |>
    mutate(prop = count / sum(count, na.rm = TRUE))
  
  p1 <- ggplot(ecoreg |> mutate(prop = 0), aes(fill = prop)) +
    geom_sf(data = countries, fill = "grey90", colour = "black") +
    geom_sf(colour = NA) +
    geom_sf(data = hex_polys |> drop_na(prop), colour = NA) +
    # geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "red", size = 0.1, shape = 3,
    #         alpha = 0.05) +
    scale_fill_viridis_c(trans = 'sqrt', option = "B") +
    coord_sf(xlim = extent[c("xmin", "xmax")],
             ylim = extent[c("ymin", "ymax")]) +
    theme_minimal()
  p1
  
  p2 <- ggplot(ecoreg |> mutate(prop = 0), aes(fill = prop)) +
    geom_sf(data = countries, fill = "grey90", colour = "black") +
    geom_sf(colour = NA) +
    geom_sf(data = hex_polys |> drop_na(prop), colour = NA) +
    geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "white", size = 0.5, 
            alpha = 0.8) +
    geom_sf(data = truth_sf, inherit.aes = FALSE, colour = "red", size = 0.02, 
           alpha = 0.8) +
    scale_fill_viridis_c(trans = 'sqrt', option = "B") +
    coord_sf(xlim = extent[c("xmin", "xmax")],
             ylim = extent[c("ymin", "ymax")]) +
    theme_minimal()
  p2
  
  p <- p1 + p2 + plot_layout(guides = "collect")
  
  list(p = p, pred_sf = pred_sf |> st_transform(proj), 
       truth_sf = truth_sf |> st_transform(proj), 
       hex_polys = hex_polys,
       ecoregions = ecoreg, 
       countries = countries)
  
}

# spec <- sample(species_df$species, 1)
# coord_predict <- generate_range_sample(spec, n = 10000, geode, env_vae, 
#                                         flow_1, scaling, geo_scaling, active_dims,
#                                         species_latent_df, squamate_train)
# map_it <- localize(coord_predict, spec, ecoregions, world)
# 
# map_it$p

for(i in 1:50) {
  spec <- sample(species_df$species, 1)
  coord_predict <- generate_range_sample(spec, n = 10000, geode, env_vae, 
                                         flow_1, scaling, geo_scaling, active_dims,
                                         species_latent_df, squamate_train)
  
  var_plots <- list()
  var_mat <- matrix(colnames(coord_predict$env_pred), nrow = 16, ncol = 2, byrow = TRUE)
  for(j in 1:nrow(var_mat)) {
    dat <- as.data.frame(coord_predict$env_pred[ , var_mat[j, ], drop = FALSE]) |>
      mutate(type = "predictions") |>
      bind_rows(as.data.frame(coord_predict$env_truth[ , var_mat[j, ], drop = FALSE]) |>
                  mutate(type = "truth"))
    var_plots[[j]] <- ggplot(dat, aes(.data[[var_mat[j, 1]]], .data[[var_mat[j, 2]]])) +
      geom_point(aes(colour = type), alpha = 0.3) +
      theme_minimal() +
      theme(axis.title = element_text(size = 6))
  }
  
  p <- wrap_plots(var_plots, nrow = 4, ncol = 4, guides = "collect") +
    plot_annotation(title = spec)
  plot(p)
  map_it <- localize(coord_predict, spec, ecoregions, world)
  pdf(file.path("output/species_tests", paste0(spec, ".pdf")))
  plot(p)
  plot(map_it$p)
  dev.off()
  plot(map_it$p)
  write_rds(list(map_it = map_it, coord_predict = coord_predict, p = p), 
            file.path("output/species_tests", paste0(spec, ".rds")))
  print(i)
}

library(rayshader)
map_it$ecoregions


p1 <- ggplot(world) +
  geom_density2d_filled(data = as.data.frame(coord_pred), aes(V1, V2), bins = 500) +
  #geom_hex(data = as.data.frame(coord_pred), aes(V1, V2), bins = c(200, 100)) +
  #geom_point(data = as.data.frame(coord_pred2), aes(V1, V2), bins = 100) +
  geom_point(aes(X, Y), data = spec_xy,
             colour = "red", alpha = 0.1, size = 0.01, shape = 3) +
  geom_sf(data = world, inherit.aes = FALSE) +
  #scale_fill_viridis_c()+#trans = "log1p") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))
p1

p2 <- ggplot(world) +
  geom_density2d_filled(data = as.data.frame(coord_pred), aes(V1, V2), bins = 500) +
  #geom_hex(data = as.data.frame(coord_pred), aes(V1, V2), bins = c(200, 100)) +
  #geom_point(data = as.data.frame(coord_pred2), aes(V1, V2), bins = 100) +
  # geom_point(aes(X, Y), data = spec_xy,
  #            colour = "red", alpha = 0.1, size = 0.01, shape = 3) +
  geom_sf(data = world, inherit.aes = FALSE) +
  #scale_fill_viridis_c()+#trans = "log1p") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))
p2

p1 + p2


# Updated function to find an appropriate equal-area projection



for(i in 1:ncol(test)) {
  rang <- range(c(spec_train[, i], pred_niche_vc[ , i]))
  plot(spec_train[,i], col = "blue", ylim = rang, main = i)
  points(pred_niche_vc[,i], col = "red", cex = 0.4)
}


ggplot(world) +
  geom_density2d_filled(data = as.data.frame(coord_pred), aes(V1, V2), bins = 500) +
  #geom_hex(data = as.data.frame(coord_pred), aes(V1, V2), bins = c(200, 100)) +
  #geom_point(data = as.data.frame(coord_pred2), aes(V1, V2), bins = 100) +
  geom_point(aes(X, Y), data = squamate_train |>
               filter(Binomial == "Varanus ornatus") |>
               select(X, Y),
             colour = "red", alpha = 1, size = 0.01) +
  geom_sf(data = world, inherit.aes = FALSE) +
  #scale_fill_viridis_c()+#trans = "log1p") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

ggplot(world) + geom_sf() + theme_classic()


ggplot(as_tibble(pred_niche_vc), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
  #geom_density2d_filled(bins = 50) +
  geom_bin_2d(bins = 50, drop = FALSE) +
  geom_point(data = squamate_train |>
               filter(Binomial == "Varanus ornatus") |>
               select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
             colour = "red", alpha = 1, size = 0.1,
             position = position_jitter(width = 0.25, height = 0.25)) +
  scale_fill_viridis_c(na.value = "black", trans = "log1p") +
  #ylim(0, 750) +
  #xlim(15, 30) +
  xlab("BIO01: Mean Annual Temperature") +
  ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(#legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

plot(env_pred[, c("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1")])
points(squamate_train |>
         filter(Binomial == "Varanus caudolineatus") |>
         select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
       col = "red", pch = 19, cex = 0.5)

plot(squamate_train |>
       filter(Binomial == "Varanus caudolineatus") |>
       select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
     col = "red", pch = 19)
plot(env_pred[ , c("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1")])

Varanus_caudolineatus <- species_latent_df |>
  filter(species == "Varanus caudolineatus")
#  filter(grepl("Varanus", species))

ggplot(species_latent_df) + 
  geom_path(alpha = 0.25, colour = "orange") + 
  geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                          L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                          L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                          L13 = 0, L14 = 0, L15 = 0, L16 = 0),
            colour = "white", linewidth = 1) +
  geom_path(data = Varanus_caudolineatus, colour = "turquoise", linewidth = 2) +
  coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                   axes.layout = "radial", scaling = 'none') +
  theme_minimal()



Varanus_mertensi <- species_latent_df |>
  filter(species == "Varanus mertensi")

latent_vm <- Varanus_mertensi |> select(starts_with("L")) |> unlist()

pred_niche_vm <- predict_niche(env_vae, flow_1, latent = latent_vm, scaling = scaling)

ggplot(as_tibble(pred_niche_vm), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
  geom_density2d_filled(bins = 50) +
  geom_point(data = squamate_train |>
               filter(Binomial == "Varanus mertensi") |>
               select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
             colour = "red", alpha = 1, size = 0.25,
             position = position_jitter(width = 0.25, height = 0.25)) +
  ylim(0, 2200) +
  xlim(17.5, 32.5) +
  xlab("BIO01: Mean Annual Temperature") +
  ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

trajs_vm <- predict_geo(geode, pred_niche_vm, geo_scaling)

coord_pred <- t((t(trajs_vm$trajectories[500, , ]) * s_xy) + mu_xy)
plot(coord_pred, col = alpha("blue", 0.01), pch = 19)
points(squamate_train |>
         filter(Binomial == "Varanus mertensi") |>
         select(X, Y),
       col = alpha("red", 0.5), cex = 0.1)

ggplot(species_latent_df) + 
  geom_path(alpha = 0.25, colour = "orange") + 
  geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                          L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                          L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                          L13 = 0, L14 = 0, L15 = 0, L16 = 0),
            colour = "white", linewidth = 1) +
  geom_path(data = Varanus_mertensi, colour = "turquoise", linewidth = 2) +
  coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                   axes.layout = "radial", scaling = 'none') +
  theme_minimal()

Sceloporus_woodi <- species_latent_df |>
  filter(species == "Sceloporus woodi")

latent_sw <- Sceloporus_woodi |> select(starts_with("L")) |> unlist()

pred_niche_sw <- predict_niche(env_vae, flow_1, latent = latent_sw, scaling = scaling)

ggplot(as_tibble(pred_niche_sw), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
  geom_density2d_filled(bins = 50) +
  geom_point(data = squamate_train |>
               filter(Binomial == "Sceloporus woodi") |>
               select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
             colour = "red", alpha = 1, size = 0.25,
             position = position_jitter(width = 0.25, height = 0.25)) +
  ylim(750, 2000) +
  xlim(17.5, 27.5) +
  xlab("BIO01: Mean Annual Temperature") +
  ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

ggplot(species_latent_df) + 
  geom_path(alpha = 0.25, colour = "orange") + 
  geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                          L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                          L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                          L13 = 0, L14 = 0, L15 = 0, L16 = 0),
            colour = "white", linewidth = 1) +
  geom_path(data = Sceloporus_woodi, colour = "turquoise", linewidth = 2) +
  coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                   axes.layout = "radial", scaling = 'none') +
  theme_minimal()

tree <- read.tree("data/phylogenies/squamates/squamates.tre")

tree_df <- pf_as_pf(tree)

# BIO1 = Annual Mean Temperature; BIO12 = Annual Precipitation

##### Latent interpolation ########
latent_1 <- latent_vc
latent_2 <- latent_sw
var_range_1 <- c(15, 30)
var_range_2 <- c(0, 2000)
interpolate_niche <- function(env_vae, flow_1, latent_1, latent_2, segs = 50, scaling,
                              vars = c("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
                              specis_latent_df,
                              var_range_1 = NULL, var_range_2 = NULL) {
  ## do endpoints first to get the plot axes scales
  n1 <- predict_niche(env_vae, flow_1, n = 10000, latent_1, scaling, device = "cuda")
  n2 <- predict_niche(env_vae, flow_1, n = 10000, latent_2, scaling, device = "cuda")
  if(is.null(var_range_1)) {
    var_range_1 <- range(c(n1[, vars[1]], n2[, vars[1]])) 
  }
  if(is.null(var_range_2)) {
    var_range_2 <- range(c(n1[, vars[2]], n2[, vars[2]]))
  }
  interps <- seq(0, 1, length.out = segs)
  interps <- interps[-1]
  interps <- interps[-length(interps)]
  vect <- latent_2 - latent_1
  vec_interp <- matrix(latent_1, nrow = length(interps), ncol = length(latent_1), byrow = TRUE) + 
    interps * matrix(vect, nrow = length(interps), ncol = length(vect), byrow = TRUE)
  
  vec_df <- as.data.frame(vec_interp)
  colnames(vec_df) <- paste0("L", 1:ncol(vec_interp))
  
  vec_1 <- as.data.frame(matrix(latent_1, nrow = 1))
  vec_2 <- as.data.frame(matrix(latent_2, nrow = 1))
  colnames(vec_1) <- paste0("L", 1:ncol(vec_interp))
  colnames(vec_2) <- paste0("L", 1:ncol(vec_interp))
  
  niches <- list()
  niche_plots <- list()
  latent_plots <- list()
  niches[[1]] <- n1
  niches[[segs]] <- n2
  niche_plots[[1]] <- ggplot(as_tibble(n1), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
    geom_density2d_filled(bins = 50) +
    ylim(var_range_2[1], var_range_2[2]) +
    xlim(var_range_1[1], var_range_1[2]) +
    xlab("BIO01: Mean Annual Temperature") +
    ylab("BIO12: Mean Annual Rainfall") +
    theme_minimal() +
    theme(legend.position = 'none',
          axis.title = element_text(size = 24),
          axis.text = element_text(size = 16))
  niche_plots[[segs]] <- ggplot(as_tibble(n2), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
    geom_density2d_filled(bins = 50) +
    ylim(var_range_2[1], var_range_2[2]) +
    xlim(var_range_1[1], var_range_1[2]) +
    xlab("BIO01: Mean Annual Temperature") +
    ylab("BIO12: Mean Annual Rainfall") +
    theme_minimal() +
    theme(legend.position = 'none',
          axis.title = element_text(size = 24),
          axis.text = element_text(size = 16))
  latent_plots[[1]] <- ggplot(species_latent_df) + 
    geom_path(alpha = 0.25, colour = "orange") + 
    geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                            L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                            L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                            L13 = 0, L14 = 0, L15 = 0, L16 = 0),
              colour = "white", linewidth = 1) +
    geom_path(data = vec_1, colour = "turquoise", linewidth = 2) +
    coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                     axes.layout = "radial", scaling = 'none') +
    theme_minimal()
  latent_plots[[segs]] <- ggplot(species_latent_df) + 
    geom_path(alpha = 0.25, colour = "orange") + 
    geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                            L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                            L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                            L13 = 0, L14 = 0, L15 = 0, L16 = 0),
              colour = "white", linewidth = 1) +
    geom_path(data = vec_2, colour = "turquoise", linewidth = 2) +
    coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                     axes.layout = "radial", scaling = 'none') +
    theme_minimal()
  for(i in 1:nrow(vec_interp)) {
    
    niches[[i + 1]] <- predict_niche(env_vae, flow_1, n = 10000, vec_interp[i, ], scaling, device = "cuda")
    
    p <- ggplot(as_tibble(niches[[i + 1]]), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
      geom_density2d_filled(bins = 50) +
      ylim(var_range_2[1], var_range_2[2]) +
      xlim(var_range_1[1], var_range_1[2]) +
      xlab("BIO01: Mean Annual Temperature") +
      ylab("BIO12: Mean Annual Rainfall") +
      theme_minimal() +
      theme(legend.position = 'none',
            axis.title = element_text(size = 24),
            axis.text = element_text(size = 16))
    p2 <- ggplot(species_latent_df) + 
      geom_path(alpha = 0.25, colour = "orange") + 
      geom_path(data = tibble(L1 = 0, L2 = 0, L3 = 0, L4 = 0,
                              L5 = 0, L6 = 0, L7 = 0, L8 = 0,
                              L9 = 0, L10 = 0, L11 = 0, L12 = 0,
                              L13 = 0, L14 = 0, L15 = 0, L16 = 0),
                colour = "white", linewidth = 1) +
      geom_path(data = vec_df[i, ], colour = "turquoise", linewidth = 2) +
      coord_serialaxes(axes.sequence = colnames(species_latent_df)[-1:-2],
                       axes.layout = "radial", scaling = 'none') +
      theme_minimal()
    
    print(p)
    niche_plots[[i + 1]] <- p
    latent_plots[[i + 1]] <- p2
    cuda_empty_cache()
    print(i)
  }
  
  list(niches = niches, niche_plots = niche_plots, latent_plots = latent_plots, n1 = n1, n2 = n2, vec_interp = vec_interp)
  
}

interp_list <- interpolate_niche(env_vae, flow_1, latent_vc, latent_sw, scaling = scaling,
                                 specis_latent_df = species_latent_df, var_range_1 = c(15, 30),
                                 var_range_2 = c(0, 2000))

write_rds(interp_list, "output/niche_interpolate_1.rds")

##### most unusual niches ##############

norms <- apply(as.matrix(latent_df)^2, 1, function(x) sqrt(sum(x)))
plot(norms)
choose <- sample(which(norms > 1.5))
spec_ch <- species_latent_df$species[choose]


##### niche UMAP ##########
write_csv(species_latent_df, "output/species_latent_df.csv")
niche_umap <- umap(latent_df, n_neighbors = 5) 

###### get geo areas #######
library(sf)
squa_sf <- read_rds("data/final_squamate_sf.rds")
summary(squa_sf$Area)
write_csv(squa_sf |> as_tibble() |> select(Binomial, Area), "output/squamate_areas.csv")




colnames(pred_niche_vc)
p1 <- ggplot(as_tibble(pred_niche_vc), aes(`CHELSA_bio15_1981-2010_V.2.1`, `CHELSA_bio8_1981-2010_V.2.1`)) +
  #geom_point(colour = "blue") +
  #geom_density2d_filled(bins = 50) +
  geom_hex(bins = 50) +
  geom_point(data = squamate_train |>
               filter(Binomial == "Varanus ornatus") |>
               select("CHELSA_bio15_1981-2010_V.2.1", "CHELSA_bio8_1981-2010_V.2.1"),
             colour = "red", alpha = 1, size = 0.5,
             position = position_jitter(width = 0.25, height = 0.25)) +
  scale_fill_viridis_c(trans = "log1p") +
  # geom_hex(data = squamate_train |>
  #              filter(Binomial == "Varanus ornatus") |>
  #              select("CHELSA_bio15_1981-2010_V.2.1", "CHELSA_bio8_1981-2010_V.2.1")
  #            ) +
  #ylim(0, 750) +
  #xlim(15, 30) +
  # xlab("BIO01: Mean Annual Temperature") +
  # ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

p2 <- ggplot(as_tibble(pred_niche_vc), aes(`CHELSA_bio15_1981-2010_V.2.1`, `CHELSA_bio8_1981-2010_V.2.1`)) +
  geom_point(colour = "blue", alpha = 0.01) +
  #geom_density2d_filled(bins = 50) +
  #geom_hex(bins = 50) +
  # geom_point(data = squamate_train |>
  #              filter(Binomial == "Varanus ornatus") |>
  #              select("CHELSA_bio15_1981-2010_V.2.1", "CHELSA_bio8_1981-2010_V.2.1"),
  #            colour = "red", alpha = 1, size = 0.1,
  #            position = position_jitter(width = 0.25, height = 0.25)) +
  geom_hex(data = squamate_train |>
               filter(Binomial == "Varanus ornatus") |>
               select("CHELSA_bio15_1981-2010_V.2.1", "CHELSA_bio8_1981-2010_V.2.1"),
             bins = 50) +
  #ylim(0, 750) +
  #xlim(15, 30) +
  # xlab("BIO01: Mean Annual Temperature") +
  # ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))
library(patchwork)
p1 + p2

na_prop <- squamate_train |>
  summarise(across(everything(), ~sum(is.na(.x)) / n()))