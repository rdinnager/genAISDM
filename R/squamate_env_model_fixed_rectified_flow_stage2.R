library(tidyverse)
library(torch)
library(deSolve)
library(dagnn)
library(unglue)
options(torch.serialization_version = 2)

codes_df <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  slice_sample(prop = 1)

batch_size <- 5e5
n_epoch <- 6e3

nbatches <- floor(nrow(codes_df) / batch_size) + 1
b <- gl(nbatches, batch_size, nrow(codes_df))

codes_df <- codes_df |>
  mutate(batch = b)

make_batch <- function(codes_df, batch_num = 1) {
  coord_dat <- codes_df |>
    filter(batch == batch_num)
  
  v <- coord_dat |>
    select(starts_with("V")) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  spec <- coord_dat |>
    select(starts_with("L")) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  list(v = v, spec = spec)  
}

bs <- unique(codes_df$batch)

batches <- map(bs, ~ make_batch(codes_df, .x), .progress = TRUE)

generate_training_data <- function(v,
                                   spec,
                                   device = "cuda",
                                   sample_fun = rnorm, ...) {
  
  with_no_grad({
  
    n <- v$size()[1]
    d <- v$size()[2]
    
    t <- torch_rand(n, 1, device = device)
    
    funs <- cbind(replicate(d, sample_fun(n, ...)))
    z <- torch_tensor(funs,
                      device = device)
    
    target <- v$to(device = device) - z
    
    coords <- z + t * target
    
  })
  
  list(coords = coords, t = t, spec = spec$to(device = device), target = target)
}

traj_net <- nn_module("TrajNet",
                      initialize = function(coord_dim, spec_dim, breadths = c(512, 256, 128),
                                            t_encode = 32L, spec_encode = 64L,
                                            device = "cuda") {
                        
                        if(length(breadths) != 3) {
                          stop("breadths should be length 3!")
                        }
                        
                        self$encode_t <- nn_linear(1L, t_encode)
                        
                        self$encode_spec <- nn_linear(spec_dim, spec_encode)
                        
                        self$unet <- nndag(input = ~ coord_dim,
                                            t_encoded = ~ t_encode,
                                            spec_encoded = ~ spec_encode,
                                            e_1 = input + t_encoded + spec_encoded ~ breadths[1],
                                            e_2 = e_1 + t_encoded + spec_encoded ~ breadths[2],
                                            e_3 = e_2 + t_encoded + spec_encoded ~ breadths[3],
                                            d_1 = e_3 + t_encoded + spec_encoded ~ breadths[3],
                                            d_2 = d_1 + e_3 + t_encoded + spec_encoded ~ breadths[2],
                                            d_3 = d_2 + e_2 + t_encoded + spec_encoded ~ breadths[1],
                                            output = d_3 + e_1 + t_encoded + spec_encoded ~ coord_dim,
                                            .act = list(nn_relu,
                                                        output = nn_identity)
                        )
                        
                        self$loss_function <- function(input, target) {
                          torch_mean((target - input)^2)
                        }
                        
                        ## wrapper to convert between torch and R format 
                        ## required by deSolve
                        self$desolve_wrapper <- function(t, y, parms, ...) {
                          with_no_grad({
                            y <- torch_tensor(matrix(y, ncol = coord_dim), 
                                              device = device)
                            t <- torch_tensor(matrix(t, ncol = 1, 
                                                     nrow = length(y) / coord_dim), 
                                              device = device)
                            y_new <- self$forward(coords = y$detach(), t = t$detach(), spec = parms$spec$detach())
                          })
                          list(as.vector(as.matrix(y_new$cpu())))
                        }
                        
                        self$sample_trajectory <- function(initial_vals,
                                                           spec_vals,
                                                           steps = 200) {
                          parms <- list(spec = spec_vals$detach())
                          init_dim <- dim(initial_vals)
                          y <- as.vector(as.matrix(initial_vals$cpu()))
                          res <- ode(y,
                                     seq(0, 1, length.out = steps),
                                     self$desolve_wrapper, 
                                     parms,
                                     method = "ode45",
                                     maxsteps = 1000)
                          times <- res[ , 1]
                          arr <- array(as.vector(res[ , -1]), 
                                       dim = c(steps, init_dim))
                          list(times = times, trajectories = arr)
                        }
                        
                      },
                      
                      forward = function(coords, t, spec) {
                        
                        t_encoded <- self$encode_t(t)
                        spec_encoded <- self$encode_spec(spec)
                        self$unet(input = coords, t_encoded = t_encoded, spec_encoded = spec_encoded) |>
                          self$unet(t_encoded = t_encoded, spec_encoded = spec_encoded)
                        
                      })

coord_dim <- 6
spec_dim <- codes_df |> select(starts_with("L")) |> slice_head(n = 1) |> ncol()

checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d"
if(!dir.exists(checkpoint_fold)) {
  dir.create(checkpoint_fold)
  trajnet <- traj_net(coord_dim, spec_dim)
  i <- 0
  start_epoch <- 0
  epoch <- 0
  batch_num <- 0
} else {
  checkpoint_files <- list.files(checkpoint_fold, full.names = TRUE, pattern = ".pt")
  checkpoints <- file.info(checkpoint_files)
  most_recent <- which.max(checkpoints$mtime)
  checkpoint <- checkpoint_files[most_recent]
  trajnet <- torch_load(checkpoint)
  progress <- unglue_data(basename(checkpoint_files),
                                   "epoch_{epoch}_batch_{batch_num}_model.pt")[most_recent, ]
  i <- 0
  start_epoch <- as.numeric(progress$epoch)
  epoch <- start_epoch
  batch_num <- 0
}
trajnet <- trajnet$cuda()
trajnet

## test trajectory
# y_init <- torch_randn(1000, coord_dim)
# spec <- torch_randn(1000, spec_dim)
# traj_test <- trajnet$sample_trajectory(y_init$cuda(), spec$cuda())
# traj_test

#plot(traj_test$trajectories[ , 100, ], type = "l")

# sample_batch <- function(train_dat, batch_num = 1) {
#   coord_dat <- train_dat |>
#     filter(batch == batch_num)
#   
#   coords <- coord_dat |>
#     select(x_interp, y_interp) |>
#     as.matrix() |>
#     torch_tensor(pin_memory = TRUE)
#   
#   t <- coord_dat |>
#     select(t) |>
#     as.matrix() |>
#     torch_tensor(pin_memory = TRUE)
#   
#   env <- coord_dat |>
#     select(-x_interp, -y_interp, -x_traj, -y_traj, -t, -batch) |>
#     as.matrix() |>
#     torch_tensor(pin_memory = TRUE)
#   
#   target <- coord_dat |>
#     select(x_traj, y_traj) |>
#     as.matrix() |>
#     torch_tensor(pin_memory = TRUE)
#   
#   list(coords = coords, t = t, env = env, target = target)  
# }

# train_dat <- generate_training_data(chelsa_df, batch_size,
#                                     x_sample_fun = x_samp,
#                                     y_sample_fun = y_samp)
# batches <- unique(train_dat$batch)
n_batches <- length(batches)
lr <- 1e-3
optimizer <- optim_adamw(trajnet$parameters, lr = lr, weight_decay = 0.01)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = n_batches,
                          cycle_momentum = FALSE)

plot_result <- function(trajnet, codes_df, n_specs = 9, steps = 250,
                        sample_fun = rnorm, ...) {
  with_no_grad({
    
    specsamp <- sample(unique(codes_df$species), n_specs)
    
    coord_dat <- codes_df |>
      filter(species %in% specsamp)
    
    spect <- coord_dat |>
      select(starts_with("L")) |>
      as.matrix() |>
      torch_tensor(pin_memory = TRUE)
    
    n <- dim(spect)[1]
    d <- coord_dat |>
      select(starts_with("V")) |>
      ncol()
    
    funs <- cbind(replicate(d, sample_fun(n)))
    y_init <- torch_tensor(funs)
    res <- trajnet$sample_trajectory(y_init$cuda(), spect$cuda(), steps = steps)
    final <- res$trajectories[steps, , ]
    colnames(final) <- paste0("T", 1:ncol(final))
    coord_dat <- coord_dat |>
      bind_cols(as.data.frame(final))
    col_pick <- map(specsamp, ~ sample.int(d, 2))
    old <- par(mfrow = c(3, 3))
    walk2(specsamp, col_pick, ~ {
        all  <- rbind(coord_dat |> filter(species == .x) |> select(all_of(paste0("T", .y))) |> as.matrix(),
                      coord_dat |> filter(species == .x) |> select(all_of(paste0("V", .y))) |> as.matrix());
        plot(all, col = "black", cex = 1.4);
        points(coord_dat |> filter(species == .x) |> select(all_of(paste0("V", .y))), col = alpha("darkgreen", 0.5), pch = 19, cex = 1.3);
        points(coord_dat |> filter(species == .x) |> select(all_of(paste0("T", .y))), col = alpha("red", 0.5), pch = 19, cex = 1.3)
      })
    par(old)
    y_init$detach()
  })
  cuda_empty_cache()
}


losses <- numeric(n_epoch * n_batches)
total_iter <- n_epoch * n_batches
epoch_times <- numeric(total_iter)

cat("Plotting current result: \n")
ragg::agg_png(file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_plot.png")),
              width = 1256, height = 1256)
try(plot_result(trajnet, codes_df))
dev.off()
torch_save(trajnet, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))

final_epoch <- start_epoch + n_epoch

for(epoch in start_epoch + seq_len(n_epoch)) {
  
  epoch_time <- Sys.time()
  
  batch_num <- 0
  for(b in batches) {
    batch_num <- batch_num + 1
    i <- i + 1
    
    optimizer$zero_grad()
    
    batch <- generate_training_data(batches[[batch_num]]$v,
                                    batches[[batch_num]]$spec,
                                    device = "cuda")
    
    output <- trajnet(batch$coords,
                      batch$t,
                      batch$spec)
    
    loss <- trajnet$loss_function(output, batch$target)
    losses[i] <- as.numeric(loss$cpu())
    
    if(batch_num %% 2 == 0) {
      
      cat("Epoch: ", epoch,
          "    batch: ", batch_num,
          "    iteration: ", i,
          "    loss: ", as.numeric(loss$cpu()),
          "\n")
      
    }
    
    if(i %% 1000 == 0) {
      cat("Plotting current result: \n")
      ragg::agg_png(file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_plot.png")),
                    width = 1256, height = 1256)
      try(plot_result(trajnet, codes_df, steps = 250))
      dev.off()
    }
    if(i %% 5000 == 0) {
      torch_save(trajnet, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))
    }
    
    loss$backward()
    optimizer$step()
    scheduler$step()
  }
  
  time <- Sys.time() - epoch_time
  epoch_times[i] <- time
  cat("Estimated time remaining: ")
  print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (final_epoch - epoch)))
}

cat("Plotting current result: \n")
ragg::agg_png(file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_plot.png")),
              width = 1256, height = 1256)
try(plot_result(trajnet, codes_df))
dev.off()
torch_save(trajnet, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))
