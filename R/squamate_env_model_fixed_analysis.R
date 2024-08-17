library(tidyverse)
library(torch)
library(ape)
library(phyf)
library(ggmulti)
library(deSolve)
library(uwot)

options(torch.serialization_version = 2)
env_vae <- torch_load("data/env_vae_1_trained_fixed.to")
env_vae <- env_vae$cuda()
flow_1 <- torch_load("output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_2/epoch_6000_batch_10_model.pt")
flow_1 <- flow_1$cuda()
#geode <- torch_load("output/checkpoints/geo_env_model_1/epoch_2862_batch_7_model.pt")
#geode <- geode$cuda()

active_dims <- read_rds("output/squamate_env_mani.rds")

spec_latents <- as.matrix(env_vae$species_embedder_mean$weight$cpu())

squamate_train <- read_csv("output/squamate_training.csv")
species_df <- tibble(id = as.integer(as.numeric(as.factor(squamate_train$Binomial))),
                     species = squamate_train$Binomial) |>
  distinct()

scaling <- read_rds("output/squamate_env_scaling.rds")
geo_scaling <- read_rds("output/chelsa_geo_scaling.rds")

latent_df <- as.data.frame(spec_latents[species_df$id, ])
colnames(latent_df) <- paste0("L", 1:16)
species_latent_df <- bind_cols(species_df, latent_df)

median_lats <- squamate_train |>
  group_by(Binomial) |>
  summarise(mid_lat = median(Y))

write_csv(median_lats, "output/squamate_latitudes.csv")

#tree <- read.tree("data/phylogenies/squamates/squamates.tre")

predict_niche <- function(env_vae, flow_1, n = 10000, latent, scaling, device = "cuda") {
  with_no_grad({
    if(is.null(dim(latent)) || dim(latent)[1] == 1) {
      latent <- matrix(as.vector(latent), nrow = n, ncol = length(latent), byrow = TRUE)
    }
    latent_tens <- torch_tensor(latent, device = device)
    samp1 <- torch_randn(n, 3L, device = device)
    samp2 <- flow_1$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 250)
    samp <- matrix(0, ncol = env_vae$spec_embed_dim, nrow = n)
    samp[ , active_dims] <- samp2$trajectories[250, , ]
    samp <- torch_tensor(samp, device = device)
    decoded <- env_vae$decoder(z = samp, s = latent_tens)
  })
  env_pred <- as.matrix(decoded$cpu())
  colnames(env_pred) <- names(scaling$means)
  env_pred <- t((t(env_pred) * scaling$sd) + scaling$means)
  env_pred
}

Varanus_caudolineatus <- species_latent_df |>
  filter(species == "Varanus caudolineatus")

Varanus_caudolineatus_train <- squamate_train |>
  filter(Binomial == "Varanus caudolineatus") |>
  select(-Binomial, -X, -Y, -n) |>
  as.matrix() |>
  scale(center = unlist(scaling$means), scale = unlist(scaling$sd))
Varanus_caudolineatus_train[is.na(Varanus_caudolineatus_train)] <- 0

latent_vc <- Varanus_caudolineatus |> select(starts_with("L")) |> unlist()
latent_vc2 <- matrix(as.vector(latent_vc), nrow = nrow(Varanus_caudolineatus_train), ncol = length(latent_vc), byrow = TRUE)

# Varanus_caudolineatus_codes <- env_vae$encoder(torch_tensor(Varanus_caudolineatus_train)$cuda(),
#                                                s = torch_tensor(latent_vc2)$cuda())
# codes <- as.matrix(Varanus_caudolineatus_codes$means$cpu())
# active_dims <- which((torch_exp(Varanus_caudolineatus_codes$logvars$mean(dim = 1L))$cpu() |> as.matrix()) < 0.5)

# plot_q <- function(x) {
#   qqnorm(x)
#   qqline(x)
# }
# 
# plot_q(codes[ , active_dims[1]])
# plot_q(codes[ , active_dims[2]])
# plot_q(codes[ , active_dims[3]])
# hist(codes[ , active_dims[1]], breaks = 30)

pred_niche_vc <- predict_niche(env_vae, flow_1, latent = latent_vc, scaling = scaling)

predict_geo <- function(geode, env_pred, geo_scaling, device = "cuda") {
  mu <- unlist(geo_scaling$mean)[-1:-2]
  s <- unlist(geo_scaling$sd[-1:-2])
  with_no_grad({
    n <- dim(env_pred)[1]
    env <- torch_tensor(t((t(env_pred) - mu) / s), device = device)
    samp <- torch_randn(n, 2) * 1.1
    trajs <- geode$sample_trajectory(initial_vals = samp, env_vals = env, steps = 500)
  })
  trajs
}

trajs <- predict_geo(geode, pred_niche_vc, geo_scaling)

mu_xy <- unlist(geo_scaling$mean[1:2])
s_xy <- unlist(geo_scaling$sd[1:2])
coord_pred <- t((t(trajs$trajectories[500, , ]) * s_xy) + mu_xy)
plot(coord_pred, col = alpha("blue", 0.01), pch = 19)
points(squamate_train |>
         filter(Binomial == "Varanus caudolineatus") |>
         select(X, Y),
       col = alpha("red", 0.5), cex = 0.1)

ggplot(as.data.frame(coord_pred), aes(V1, V2)) +
  geom_density2d_filled() +
  geom_point(aes(X, Y), data = squamate_train |>
               filter(Binomial == "Varanus caudolineatus") |>
               select(X, Y),
             colour = "red", alpha = 0.05) +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.title = element_text(size = 24),
        axis.text = element_text(size = 16))

# true_dat <- squamate_train |>
#   filter(Binomial == "Varanus caudolineatus") |>
#   select(-Binomial, -X, -Y, -n) |>
#   as.matrix()
# true_dat[is.na(true_dat)] <- 0
# true_dat <- do.call(rbind, rep(list(true_dat), 10))
# trajs <- predict_geo(geode, true_dat, geo_scaling)

mu_xy <- unlist(geo_scaling$mean[1:2])
s_xy <- unlist(geo_scaling$sd[1:2])
coord_pred <- t((t(trajs$trajectories[500, , ]) * s_xy) + mu_xy)
plot(coord_pred, col = alpha("blue", 0.01), pch = 19)
points(squamate_train |>
         filter(Binomial == "Varanus caudolineatus") |>
         select(X, Y),
       col = alpha("red", 0.5), cex = 0.1)


ggplot(as_tibble(pred_niche_vc), aes(`CHELSA_bio1_1981-2010_V.2.1`, `CHELSA_bio12_1981-2010_V.2.1`)) +
  geom_density2d_filled(bins = 50) +
  #geom_hex(bins = 50) +
  geom_point(data = squamate_train |>
               filter(Binomial == "Varanus caudolineatus") |>
               select("CHELSA_bio1_1981-2010_V.2.1", "CHELSA_bio12_1981-2010_V.2.1"),
             colour = "red", alpha = 1, size = 0.25,
             position = position_jitter(width = 0.25, height = 0.25)) +
  ylim(0, 750) +
  xlim(15, 30) +
  xlab("BIO01: Mean Annual Temperature") +
  ylab("BIO12: Mean Annual Rainfall") +
  theme_minimal() +
  theme(legend.position = 'none',
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
  