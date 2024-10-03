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
library(data.table)
library(unglue)
library(torchopt)

source("R/load_and run_models.R")
source("R/geo_utils_and_data.R")
source("R/plotting.R")

training_data <- load_training_data()
test_data <- load_test_data()
polygons <- load_species_polygons()

distill_data <- fread("output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d_distill_data/all_zs.csv")

dat <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  distinct(species, .keep_all = TRUE)
spec_lats <- dat |>
  select(species, starts_with("L"))

sds <- spec_lats |>
  summarise(across(everything(), ~sd(.x, na.rm = TRUE)))
msd <- mean(unlist(sds), na.rm = TRUE)

distill_data <- distill_data |>
  left_join(spec_lats)

squamate_train <- read_csv("output/squamate_training2.csv")
squamate_test <- read_csv("output/squamate_testing2.csv")

test_specs <- unique(squamate_test$Binomial)
test_specs <- test_specs[!test_specs %in% squamate_train$Binomial]

test_df <- squamate_test |>
  filter(Binomial %in% test_specs) |>
  group_by(Binomial) |>
  slice_sample(n = 100) |>
  ungroup()

metadata <- read_rds("data/final_squamate_sf.rds")
test_spec_df <- test_df |>
  distinct(Binomial, .keep_all = TRUE) |>
  left_join(metadata)

scaling <- read_rds("output/squamate_env_scaling2.rds")
geo_scaling <- read_rds("output/chelsa_geo_scaling.rds")

# checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_distill_7d"
# 
# if(!dir.exists(checkpoint_fold)) {
#   dir.create(checkpoint_fold)
#   checkpoint_dir <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d"
#   files <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
#   file_info <- file.info(files)
#   latest <- which.max(file_info$mtime)
#   mod_file <- files[latest]
#   flow_2 <- torch_load(mod_file)
#   i <- 0
#   start_epoch <- 0
#   epoch <- 0
#   batch_num <- 0
# } else {
#   checkpoint_files <- list.files(checkpoint_fold, full.names = TRUE, pattern = ".pt")
#   checkpoints <- file.info(checkpoint_files)
#   most_recent <- which.max(checkpoints$mtime)
#   checkpoint <- checkpoint_files[most_recent]
#   flow_2 <- torch_load(checkpoint)
#   progress <- unglue_data(basename(checkpoint_files),
#                           "epoch_{epoch}_batch_{batch_num}_model.pt")[most_recent, ]
#   i <- 0
#   start_epoch <- as.numeric(progress$epoch)
#   epoch <- start_epoch
#   batch_num <- 0
# }
flow_2 <- load_nichencoder_flow_rectified()
#flow_2 <- flow_2$cuda()
flow_2

spec_samp <- sample(unique(distill_data$species), 9)
spec_df <- distill_data |>
  filter(species %in% spec_samp)
spec_l <- spec_df |>
  select(starts_with("L")) |>
  as.matrix() |>
  torch_tensor()
zs <- spec_df |>
  select(starts_with("Z")) |>
  as.matrix() |>
  torch_tensor()

test1 <- flow_2$sample_trajectory(zs$cuda(), spec_l$cuda(), steps = 2)

vs_pr <- test1$trajectories[2, , ] |>
  as.data.frame() |>
  mutate(type = "predictions", species = spec_df$species) |>
  bind_rows(spec_df |>
              select(starts_with("V")) |>
              mutate(type = "truth", species = spec_df$species))

ggplot(vs_pr, aes(V1, V2)) +
  geom_point(aes(colour = type), alpha = 0.2) +
  facet_wrap(vars(species), nrow = 3) +
  theme_minimal()

test2 <- zs$cuda() + flow_2(zs$cuda(), t = torch_zeros(zs$size()[[1]], 1, device = "cuda"), spec_l$cuda())
vs_pr2 <- as.matrix(test2$cpu()) |>
  as.data.frame() |>
  mutate(type = "predictions", species = spec_df$species) |>
  bind_rows(spec_df |>
              select(starts_with("V")) |>
              mutate(type = "truth", species = spec_df$species)) |>
  slice_sample(prop = 1)

ggplot(vs_pr2, aes(V1, V2)) +
  geom_point(aes(colour = type), alpha = 0.2) +
  facet_wrap(vars(species), nrow = 3) +
  theme_minimal()

# checkpoint_dir_geo <- "output/checkpoints/geo_env_model_3"
# files_geo <- list.files(checkpoint_dir_geo, full.names = TRUE, pattern = ".pt")
# file_info_geo <- file.info(files_geo)
# latest_geo <- which.max(file_info_geo$mtime)
# mod_file_geo <- files_geo[latest_geo]

#env_vae <- torch_load("data/env_vae_trained_fixed2_alpha_0.5_32d.to")
env_vae <- load_nichencoder_vae()
#env_vae <- env_vae$cuda()

options(torch.serialization_version = 3)
# geode <- torch_load(mod_file_geo)
# geode <- geode$cuda()
geode <- load_geode_flow()


active_dims <- read_rds("output/squamate_env_mani2.rds")

# Differentiable Soft Sorting Function
soft_sort <- function(x, tau = 1.0, dim = -1) {
  # x: Input tensor
  # tau: Temperature parameter controlling the approximation smoothness
  # dim: Dimension along which to sort
  
  # Compute the pairwise differences
  x_expand_1 <- x$unsqueeze(dim + 1)  # Shape: [..., n, 1]
  x_expand_2 <- x$unsqueeze(dim)      # Shape: [..., 1, n]
  
  # Compute pairwise differences
  diff = x_expand_1 - x_expand_2  # Shape: [..., n, n]
  
  # Compute the softmax weights
  P = nnf_softmax(-diff / tau, dim = dim + 1)  # Softmax over the last dimension
  
  # Compute expected sorted values
  x_sorted = P$matmul(x$unsqueeze(dim + 1))  # Shape: [..., n, 1]
  x_sorted = x_sorted$squeeze(dim + 1)       # Shape: [..., n]
  
  return(x_sorted)
}


zeroshot_nn <- nn_module("ZeroShotNN",
                         initialize = function(n_mc = 100, bandwidth = 1) {
                           self$spec_l <- nn_parameter(torch_randn(1, 32) * msd)
                           self$n_mc <- n_mc
                           self$bandwidth <- bandwidth
                         },
                         forward = function(n = NULL) {
                           if(is.null(n)) {
                             n <- self$n_mc 
                           }
                           spec_l_mat <- self$spec_l$`repeat`(c(n, 1L))
                           zs <- torch_randn(n, 6L, device = "cuda")
                           
                           vae_lat <- zs + flow_2(zs, t = torch_zeros(zs$size()[[1]], 1, device = "cuda"), spec_l_mat)
                           
                           samp <- torch_zeros(nrow = vae_lat$size()[[1]], env_vae$latent_dim, device = "cuda")
                           samp[ , active_dims] <- vae_lat
                           
                           env_pred <- env_vae$decoder(z = samp, s = spec_l_mat)
                           
                           return(env_pred)
                         },
                         sliced_wasserstein_distance = function(X, Y, num_projections = 100, p = 1) {
                           # X and Y are torch tensors of shape (n_samples, n_features)
                           # num_projections: Number of random projections
                           # p: Order of the Wasserstein distance (usually p = 1 or p = 2)
                           n_features <- X$size()[2]
                           
                           # Generate random projection vectors (unit vectors)
                           projections <- torch_randn(num_projections, n_features, device = "cuda")
                           projections <- projections / projections$norm(dim = 2, keepdim = TRUE)
                           
                           # Project the data
                           proj_X <- X$matmul(projections$t())  # Shape: (n_samples_X, num_projections)
                           proj_Y <- Y$matmul(projections$t())  # Shape: (n_samples_Y, num_projections)
                           
                           proj_X_sorted <- soft_sort(proj_X, tau = tau, dim = 1)
                           proj_Y_sorted <- soft_sort(proj_Y, tau = tau, dim = 1)
                           
                           # Compute the Wasserstein distances for each projection
                           distances <- (proj_X_sorted - proj_Y_sorted)$abs()$pow(p)
                           
                           # Average over projections and samples
                           swd <- distances$mean()$pow(1.0 / p)
                           
                           return(swd)
                         },
                         
                         sinkhorn_loss = function(x, y, epsilon = 0.05, nx = 100, ny = 100, niter = 50, p = 1) {
                           # x: Tensor of shape (n, d)
                           # y: Tensor of shape (n, d)
                           # epsilon: Regularization parameter
                           # n: Number of samples
                           # niter: Number of iterations
                           device <- x$device
                           
                           # Compute the cost matrix
                           C <- torch_cdist(x, y, p = 2)$pow(p)  # Shape: (n, n)
                           
                           # Initialize uniform marginal distributions
                           mu <- torch_full(c(nx), 1.0 / nx, dtype = torch_float(), requires_grad = FALSE, device = device)
                           nu <- torch_full(c(ny), 1.0 / ny, dtype = torch_float(), requires_grad = FALSE, device = device)
                           
                           # Parameters
                           rho <- 1.0    # Unbalanced transport parameter (can be adjusted)
                           tau <- -0.8   # Nesterov-like acceleration parameter
                           lam <- rho / (rho + epsilon)
                           thresh <- 1e-3  # Stopping criterion
                           
                           # Define helper functions
                           ave <- function(u, u1) {
                             # Barycenter subroutine for kinetic acceleration
                             tau * u + (1 - tau) * u1
                           }
                           
                           M <- function(u, v) {
                             # Modified cost for logarithmic updates
                             # u: Tensor of shape (n)
                             # v: Tensor of shape (n)
                             # Returns a tensor of shape (n, n)
                             (-C + u$unsqueeze(2) + v$unsqueeze(1)) / epsilon
                           }
                           
                           lse <- function(A) {
                             # Log-sum-exp function
                             # A: Tensor of shape (n, n)
                             # Returns a tensor of shape (n, 1)
                             torch_logsumexp(A, dim = 2, keepdim = TRUE)
                           }
                           
                           # Initialize dual variables
                           u <- torch_zeros_like(mu)  # Shape: (n)
                           v <- torch_zeros_like(nu)  # Shape: (n)
                           err <- torch_tensor(0.0, dtype = torch_float())
                           
                           # Sinkhorn iterations
                           for (i in 1:niter) {
                             u1 <- u$clone()  # Save previous u
                             
                             # Compute M(u, v)
                             #M_uv <- M(u, v)  # Shape: (nx, ny)
                             
                             # Update u
                             u <- epsilon * (torch_log(mu) - lse(M(u, v))$squeeze()) + u  # Shape: (nx)
                             
                             # Update v
                             v <- epsilon * (torch_log(nu) - lse(M(u, v)$t())$squeeze()) + v  # Shape: (ny)
                             
                             # Compute error
                             err <- torch_sum(torch_abs(u - u1))
                             
                             # Check stopping criterion
                             if (err$item() < thresh) {
                               break
                             }
                           }
                           
                           # Compute the transport plan
                           pi <- torch_exp(M(u, v))  # Shape: (nx, ny)
                           
                           # Compute the Sinkhorn cost
                           cost <- torch_sum(pi * C)
                           
                           return(cost)
                         },
                         
                         sinkhorn_normalized = function(x, y, epsilon = 0.05, nx = 100, ny = 100, niter = 50, p = 2) {
                           Wxy <- self$sinkhorn_loss(x, y, epsilon, nx, ny, niter, p)
                           Wxx <- self$sinkhorn_loss(x, x, epsilon, nx, ny, niter, p)
                           Wyy <- self$sinkhorn_loss(y, y, epsilon, nx, ny, niter, p)
                           return(2 * Wxy - Wxx - Wyy)
                         },
                         
                         loss_function = function(input, target) {
                           
                           E_target <- torch_mean(torch_cdist(target, target))
                           E_input <- torch_mean(torch_cdist(input, input))
                           E_tar_inp <- torch_mean(torch_cdist(target, input)) 
                           2 * E_tar_inp - E_input - E_target
                         })

spec <- sample(test_df$Binomial, 1)
model_file <- "output/checkpoints/test_model_1.pt"
zeroshot_predict <- function(spec, n_mc = 100, model_file = tempfile(fileext = ".pt"),
                             stage_iters = list(c(50, 1000), c(5, 4000), 
                                                c(5, 4000), c(5, 4000), 10000)) {
  
  trajectory <- list()
  
  spec_env <- test_df |>
    filter(Binomial == spec) |>
    select(starts_with("CHELSA"))
  
  prop_nas <- spec_env |>
    summarise(across(everything(), ~sum(is.na(.x))/n()))
  probls <- colnames(prop_nas)[prop_nas > 0.5]
  
  spec_env <- spec_env |> as.matrix()
  
  if(length(probls) > 0) {
    for(i in probls) {
      spec_env[ , i] <- scaling$mean[i][[1]]  
    }
    for(i in probls) {
      spec_env[ , i] <- scaling$mean[i][[1]]  
    }
  }
  
  spec_env <- t((t(spec_env) - unlist(scaling$means)) / unlist(scaling$sd))
  spec_env <- apply(spec_env, 2, function(x) {x[is.na(x)] <- mean(x, na.rm = TRUE); x})
  spec_env_tens <- torch_tensor(spec_env, device = "cuda")
  
  best_loss <- 9999
  best_lat <- NULL
  best_lr <- NULL
  
  #spec_l <- torch_randn(1, 32L, device = "cuda", requires_grad = TRUE)
  trajectory[["stage_1"]] <- list()
  for(j in 1:stage_iters[[1]][1]) {
    
    trajectory[["stage_1"]][[j]] <- matrix(NA, ncol = 32L, nrow = stage_iters[[1]][2])
        
    zs_nn <- zeroshot_nn()
    zs_nn <- zs_nn$cuda()
    
    #lr <- sample(c(1e-4, 1e-3, 1e-2, 1e-5, 5e-4, 5e-3, 5e-2, 5e-5), 1)
    lr <- 1e-3
    n_epoch <- 1000
    optimizer <- optim_adamw(zs_nn$parameters, lr = lr, weight_decay = 0.01)
    # scheduler <- lr_one_cycle(optimizer, max_lr = lr,
    #                           epochs = n_epoch, steps_per_epoch = 1,
    #                           cycle_momentum = TRUE, pct_start = 0.3)
    
    for(epoch in seq_len(stage_iters[[1]][2])) {
      env_pred <- zs_nn()
      loss <- zs_nn$loss_function(env_pred, spec_env_tens)
      loss_save <- as.numeric(loss$cpu()$detach())
      if(loss_save < best_loss) {
        best_loss <- loss_save
        best_lat <- as.numeric(zs_nn$spec_l$cpu()$detach())
        best_lr <- lr
      }
      trajectory[["stage_1"]][[j]][epoch, ] <- as.numeric(zs_nn$spec_l$cpu()$detach())
      
      cat("Stage 1 Energy - ",
          "i: ", j,
          "Epoch: ", epoch,
          "    loss: ", as.numeric(loss$cpu()),
          "    best loss: ", best_loss,
          "\n")
      
      loss$backward()
      optimizer$step()
      
      
#      scheduler$step()
      
    }
    with_no_grad({
      zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    })
    torch_save(zs_nn, model_file)
    env_pred <- zs_nn(1000)
    
    samp_vars <- sample.int(32, 2)
    plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
    points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
    points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
  }
  
  trajectory[["stage_2"]] <- list()
  for(j in 1:stage_iters[[2]][1]) {
    
    trajectory[["stage_2"]][[j]] <- matrix(NA, ncol = 32L, nrow = stage_iters[[2]][2])
    
    with_no_grad({
      zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    })
    
    lr <- 1e-4
    n_epoch <- 4000
    optimizer <- optim_adamw(zs_nn$parameters, lr = lr, weight_decay = 0.01)
    scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                              epochs = n_epoch, steps_per_epoch = 1,
                              cycle_momentum = TRUE, pct_start = 0.3)
    
    for(epoch in seq_len(stage_iters[[2]][2])) {
      env_pred <- zs_nn()
      loss <- zs_nn$loss_function(env_pred, spec_env_tens)
      loss_save <- as.numeric(loss$cpu()$detach())
      if(loss_save < best_loss) {
        best_loss <- loss_save
        best_lat <- as.numeric(zs_nn$spec_l$cpu()$detach())
        best_lr <- lr
      }
      trajectory[["stage_2"]][[j]][epoch, ] <- as.numeric(zs_nn$spec_l$cpu()$detach())
      
      cat("Stage 2 Energy - ",
          "i: ", j,
          "Epoch: ", epoch,
          "    loss: ", as.numeric(loss$cpu()),
          "    best loss: ", best_loss,
          "\n")
      
      loss$backward()
      optimizer$step()
      scheduler$step()
      
      if(epoch %% 1000 == 0) {
        env_pred <- zs_nn(1000)
        samp_vars <- sample.int(32, 2)
        plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
        points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
        points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
      }
      
    }
    # with_no_grad({
    #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    # })
    torch_save(zs_nn, model_file)
    env_pred <- zs_nn(1000)
    
    samp_vars <- sample.int(32, 2)
    plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
    points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
    points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
  }
  
  best_loss <- 9999
  trajectory[["stage_3"]] <- list()
  
  for(j in 1:stage_iters[[3]][1]) {
    
    trajectory[["stage_3"]][[j]] <- matrix(NA, ncol = 32L, nrow = stage_iters[[3]][2])
    
    with_no_grad({
      zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    })
    
    lr <- 1e-5
    n_epoch <- 4000
    optimizer <- optim_adamw(zs_nn$parameters, lr = lr, weight_decay = 0.01)
    scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                              epochs = n_epoch, steps_per_epoch = 1,
                              cycle_momentum = TRUE, pct_start = 0.3)
    
    for(epoch in seq_len(stage_iters[[3]][2])) {
      #message("Sampling...")
      env_pred <- zs_nn()
      #message("Calculating loss...")
      loss <- 0.5 * zs_nn$sinkhorn_loss(env_pred, spec_env_tens) + 0.5 * zs_nn$loss_function(env_pred, spec_env_tens)
      #message("Saving loss...")
      loss_save <- as.numeric(loss$cpu()$detach())
      #message("Saving latent vector...")
      if(loss_save < best_loss) {
        best_loss <- loss_save
        best_lat <- as.numeric(zs_nn$spec_l$cpu()$detach())
        best_lr <- lr
      }
      trajectory[["stage_3"]][[j]][epoch, ] <- as.numeric(zs_nn$spec_l$cpu()$detach())
      
      cat("Stage 3 Energy + Sinkhorn - ",
          "i: ", j,
          "Epoch: ", epoch,
          "    loss: ", as.numeric(loss$cpu()),
          "    best loss: ", best_loss,
          "\n")
      
      #message("Gradient calculation...")
      loss$backward()
      #message("Optimizing...")
      optimizer$step()
      #message("LR Scheduling...")
      scheduler$step()
      
      if(epoch %% 1000 == 0) {
        #message("Plotting...")
        env_pred <- zs_nn(1000)
        samp_vars <- sample.int(32, 2)
        plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
        points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
        points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
      }
      
    }
    # with_no_grad({
    #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    # })
    torch_save(zs_nn, model_file)
    env_pred <- zs_nn(1000)
    
    samp_vars <- sample.int(32, 2)
    plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
    points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
    points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
  }
  
  trajectory[["stage_4"]] <- list()
  best_loss <- 9999
  
  for(j in 1:stage_iters[[4]][1]) {
    
    trajectory[["stage_4"]][[j]] <- matrix(NA, ncol = 32L, nrow = stage_iters[[4]][2])
    
    # with_no_grad({
    #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    # })
    
    lr <- 1e-6
    n_epoch <- 4000
    optimizer <- optim_adamw(zs_nn$parameters, lr = lr, weight_decay = 0.01)
    scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                              epochs = n_epoch, steps_per_epoch = 1,
                              cycle_momentum = TRUE, pct_start = 0.3)
    
    for(epoch in seq_len(stage_iters[[4]][2])) {
      env_pred <- zs_nn()
      loss <- zs_nn$sinkhorn_loss(env_pred, spec_env_tens) 
      loss_save <- as.numeric(loss$cpu()$detach())
      if(loss_save < best_loss) {
        best_loss <- loss_save
        best_lat <- as.numeric(zs_nn$spec_l$cpu()$detach())
        best_lr <- lr
      }
      trajectory[["stage_4"]][[j]][epoch, ] <- as.numeric(zs_nn$spec_l$cpu()$detach())
      
      cat("Stage 4 Sinkhorn - ",
          "i: ", j,
          "Epoch: ", epoch,
          "    loss: ", as.numeric(loss$cpu()),
          "    best loss: ", best_loss,
          "\n")
      
      loss$backward()
      optimizer$step()
      scheduler$step()
      
      if(epoch %% 1000 == 0) {
        env_pred <- zs_nn(1000)
        samp_vars <- sample.int(32, 2)
        plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
        points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
        points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
      }
      
    }
    # with_no_grad({
    #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
    # })
    torch_save(zs_nn, model_file)
    env_pred <- zs_nn(1000)
    
    samp_vars <- sample.int(32, 2)
    plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
    points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
    points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
  }
  
  # with_no_grad({
  #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
  # })
  lr <- 1e-7
  n_epoch <- 10000
  optimizer <- optim_adamw(zs_nn$parameters, lr = lr, weight_decay = 0.01)
  scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                            epochs = n_epoch, steps_per_epoch = 1,
                            cycle_momentum = TRUE, pct_start = 0.3)
  
  trajectory[["stage_5"]] <- list()
  trajectory[["stage_5"]][[1]] <- matrix(NA, ncol = 32L, nrow = stage_iters[[5]])
  
  for(epoch in seq_len(stage_iters[[5]])) {
    env_pred <- zs_nn()
    loss <- zs_nn$sinkhorn_loss(env_pred, spec_env_tens) 
    loss_save <- as.numeric(loss$cpu()$detach())
    if(loss_save < best_loss) {
      best_loss <- loss_save
      best_lat <- as.numeric(zs_nn$spec_l$cpu()$detach())
      best_lr <- lr
    }
    trajectory[["stage_5"]][[1]][epoch, ] <- as.numeric(zs_nn$spec_l$cpu()$detach())
    
    cat("Stage 5 Sinkhorn - ",
        "i: ", j,
        "Epoch: ", epoch,
        "    loss: ", as.numeric(loss$cpu()),
        "    best loss: ", best_loss,
        "\n")
    
    loss$backward()
    optimizer$step()
    scheduler$step()
    
    if(epoch %% 1000 == 0) {
      env_pred <- zs_nn(1000)
      samp_vars <- sample.int(32, 2)
      plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
      points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
      points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
    }
    
  }
  # with_no_grad({
  #   zs_nn$spec_l <- nn_parameter(torch_tensor(matrix(best_lat, nrow = 1), device = "cuda", requires_grad = TRUE))  
  # })
  torch_save(zs_nn, model_file)
  env_pred <- zs_nn(1000)
  
  samp_vars <- sample.int(32, 2)
  plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , samp_vars], col = "black", pch = 19)
  points(as.matrix(env_pred$cpu())[ , samp_vars], col = "green", pch = 19)
  points(as.matrix(spec_env_tens$cpu())[ , samp_vars], col = "red", pch = 19)
  
  list(species = spec, zs_nn = zs_nn, model_file = model_file, 
       trajectory = trajectory, env_truth = as.matrix(spec_env_tens$cpu()),
       final_latent = as.numeric(zs_nn$spec_l$cpu()$detach()),
       best_latent = best_lat)
}

###### Run on sample of species ##############
test_spec_df <- test_spec_df |>
  filter(Area > 2500, Area < 2e+6) |>
  mutate(area_cat = cut_interval(log(Area + 1), 3))

test_spec_sample <- test_spec_df |>
  group_by(area_cat) |>
  slice_sample(n = 100) |>
  ungroup() |>
  slice_sample(prop = 1)

species_name <- test_spec_sample$Binomial[2]
run_zeroshot <- function(species_name, output_dir = "output/model_results_zeroshot") {
  
  model_file <- file.path(output_dir, paste0(species_name, ".pt"))
  data_file  <- file.path(output_dir, paste0(species_name, "_data.pt"))
  env_plot_file  <- file.path(output_dir, paste0(species_name, "_env_plot.png"))
  geo_plot_file  <- file.path(output_dir, paste0(species_name, "_geo_plot.png"))
  
  if(file.exists(model_file)) {
    return(invisible(NULL))
  }
  
  try_it <- try({zeroshot_spec <- zeroshot_predict(species_name,
                                    model_file = model_file,
                                    stage_iters = list(c(50, 1000), c(3, 4000), 
                                                       c(3, 4000), c(3, 4000), 5000))
    zeroshot_spec$zs_nn <- torch_serialize(zeroshot_spec$zs_nn)
    #write_rds(zeroshot_spec, data_file)
    spec_dat <- get_species_data(species_name, training_data = test_data, test_data = test_data, polygons = polygons)
    spec_dat2 <- spec_dat
    spec_dat2$latent <- matrix(zeroshot_spec$final_latent, nrow = 1)
    problem_vars <- c("CHELSA_fcf_1981-2010_V.2.1", "CHELSA_swe_1981-2010_V.2.1")
    
    predicts <- make_model_predictions(spec_dat2, env_vae, flow_2, geode, problem_vars = problem_vars)
    
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
    plot(geo_plot$combined_test) 
    dev.off()
    
    write_rds(list(zeroshot = zeroshot_spec,
                   spec_dat = spec_dat2, predict_dat = predicts,
                   plot_data = list(env_plot = env_plot,
                                    geo_plots = geo_plot)),
              data_file)
    
  })
  
  invisible(NULL)
}

walk(test_spec_sample$Binomial, run_zeroshot,
      .progress = TRUE)



zeroshot_spec <- zeroshot_predict(spec)
zeroshot_spec$zs_nn <- torch_serialize(zeroshot_spec$zs_nn)
write_rds(zeroshot_spec, "output/test_spec_zeroshot.rds")

plot(zeroshot_spec$trajectory$stage_1[[1]][,1:2], type = "l")
points(zeroshot_spec$trajectory$stage_1[[2]][,1:2], type = "l")
plot(zeroshot_spec$trajectory$stage_2[[1]][,1:2], type = "l")

env_pred <- zeroshot_spec$zs_nn()

plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , 1:2], col = "black", pch = 19)
points(as.matrix(env_pred$cpu())[ , 1:2], col = "green", pch = 19)
points(as.matrix(spec_env_tens$cpu())[ , 1:2], col = "red", pch = 19)

plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , 3:4], col = "black", pch = 19)
points(as.matrix(env_pred$cpu())[ , 3:4], col = "green", pch = 19)
points(as.matrix(spec_env_tens$cpu())[ , 3:4], col = "red", pch = 19)

plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , 5:6], col = "black", pch = 19)
points(as.matrix(env_pred$cpu())[ , 5:6], col = "green", pch = 19)
points(as.matrix(spec_env_tens$cpu())[ , 5:6], col = "red", pch = 19)

plot(rbind(as.matrix(env_pred$cpu()), as.matrix(spec_env_tens$cpu()))[ , 7:8], col = "black", pch = 19)
points(as.matrix(env_pred$cpu())[ , 7:8], col = "green", pch = 19)
points(as.matrix(spec_env_tens$cpu())[ , 7:8], col = "red", pch = 19)

########## testing ##########


# Define point clouds
X_data <- matrix(c(0, 1), ncol = 1)
Y_data <- matrix(c(0, 2), ncol = 1)

x <- torch_tensor(X_data, device = "cuda")
y <- torch_tensor(Y_data, device = "cuda")

sinkhorn_loss = function(x, y, epsilon = 0.05, nx = 100, ny = 100, niter = 50, p = 1) {
  # x: Tensor of shape (n, d)
  # y: Tensor of shape (n, d)
  # epsilon: Regularization parameter
  # n: Number of samples
  # niter: Number of iterations
  device <- x$device
  nx <- x$size()[1]
  ny <- y$size()[1]
  
  # Compute the cost matrix
  C <- torch_cdist(x, y, p = 2)$pow(p)  # Shape: (n, n)
  
  # Initialize uniform marginal distributions
  mu <- torch_full(c(nx), 1.0 / nx, dtype = torch_float(), requires_grad = FALSE, device = device)
  nu <- torch_full(c(ny), 1.0 / ny, dtype = torch_float(), requires_grad = FALSE, device = device)
  
  # Parameters
  rho <- 1.0    # Unbalanced transport parameter (can be adjusted)
  tau <- -0.8   # Nesterov-like acceleration parameter
  lam <- rho / (rho + epsilon)
  thresh <- 1e-9  # Stopping criterion
  
  # Define helper functions
  ave <- function(u, u1) {
    # Barycenter subroutine for kinetic acceleration
    tau * u + (1 - tau) * u1
  }
  
  M <- function(u, v) {
    # Modified cost for logarithmic updates
    # u: Tensor of shape (n)
    # v: Tensor of shape (n)
    # Returns a tensor of shape (n, n)
    (-C + u$unsqueeze(2) + v$unsqueeze(1)) / epsilon
  }
  
  lse <- function(A) {
    # Log-sum-exp function
    # A: Tensor of shape (n, n)
    # Returns a tensor of shape (n, 1)
    torch_logsumexp(A, dim = 2, keepdim = TRUE)
  }
  
  # Initialize dual variables
  u <- torch_zeros_like(mu)  # Shape: (n)
  v <- torch_zeros_like(nu)  # Shape: (n)
  err <- torch_tensor(0.0, dtype = torch_float())
  
  # Sinkhorn iterations
  for (i in 1:niter) {
    u1 <- u$clone()  # Save previous u
    
    # Compute M(u, v)
    #M_uv <- M(u, v)  # Shape: (nx, ny)
    
    # Update u
    u <- epsilon * (torch_log(mu) - lse(M(u, v))$squeeze()) + u  # Shape: (nx)
    
    # Update v
    v <- epsilon * (torch_log(nu) - lse(M(u, v)$t())$squeeze()) + v  # Shape: (ny)
    
    # Compute error
    err <- torch_sum(torch_abs(u - u1))
    # cat("Iteration:", i, "\n")
    # cat("Error:", err$item(), "\n")
    # cat("u:", u$cpu() |> as.matrix(), "\n")
    # cat("v:", v$cpu() |> as.matrix(), "\n")
    # cat("Max M_uv:", torch_max(M_uv)$item(), "\n")
    # cat("Min M_uv:", torch_min(M_uv)$item(), "\n")
    
    # Check stopping criterion
    if (err$item() < thresh) {
      break
    }
  }
  
  # Compute the transport plan
  pi <- torch_exp(M(u, v))  # Shape: (nx, ny)
  
  # Compute the Sinkhorn cost
  cost <- torch_sum(pi * C)
  
  return(cost)
}

sinkhorn_normalized = function(x, y, epsilon = 0.05, nx = 100, ny = 100, niter = 50, p = 1) {
  Wxy <- sinkhorn_loss(x, y, epsilon, nx, ny, niter, p)
  Wxx <- sinkhorn_loss(x, x, epsilon, nx, ny, niter, p)
  Wyy <- sinkhorn_loss(y, y, epsilon, nx, ny, niter, p)
  return(2 * Wxy - Wxx - Wyy)
}

sinkhorn_normalized(X, Y, nx = 2, ny = 2, epsilon = 0.01, niter = 100)
sinkhorn_loss(X, X, nx = 2, ny = 2, epsilon = 0.01, niter = 100)
sinkhorn_loss(Y, Y, nx = 2, ny = 2, epsilon = 0.01)

sinkhorn_loss(X, Y, nx = 2, ny = 2, epsilon = 0.01)









