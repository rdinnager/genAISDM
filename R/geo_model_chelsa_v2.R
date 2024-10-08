library(tidyverse)
library(torch)
library(deSolve)
library(dagnn)
library(unglue)
options(torch.serialization_version = 3)

chelsa_df <- read_rds("output/geo_env_chelsa.rds")

## Remove problematic environmental variables
problem_vars <- c("CHELSA_fcf_1981-2010_V.2.1", "CHELSA_swe_1981-2010_V.2.1")
chelsa_df <- chelsa_df |>
  select(-all_of(problem_vars))

scaling <- list(mean = chelsa_df |>
                  summarise(across(everything(), ~ mean(.x, na.rm = TRUE))),
  sd = chelsa_df |>
    summarise(across(everything(), ~ sd(.x, na.rm = TRUE))))

write_rds(scaling, "output/chelsa_geo_scaling_v2.rds")

chelsa_df <- chelsa_df |>
  mutate(across(everything(), ~ (.x - mean(.x, na.rm = TRUE)) / sd(.x, na.rm = TRUE)))

## randomize and fill in NAs
chelsa_df <- chelsa_df |>
  slice_sample(prop = 1) |>
  mutate(across(everything(), ~ replace_na(.x, 0)))

batch_size <- 1.5e6
n_epoch <- 5e3

nbatches <- floor(nrow(chelsa_df) / batch_size) + 1
b <- gl(nbatches, batch_size, nrow(chelsa_df))

chelsa_df <- chelsa_df |>
  mutate(batch = b)

make_batch <- function(chelsa_df, batch_num = 1) {
  coord_dat <- chelsa_df |>
    filter(batch == batch_num)
  
  xy <- coord_dat |>
    select(x, y) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  env <- coord_dat |>
    select(-x, -y, -batch) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  list(xy = xy, env = env)  
}

bs <- unique(chelsa_df$batch)

batches <- map(bs, ~ make_batch(chelsa_df, .x), .progress = TRUE)

generate_training_data <- function(xy,
                                   env,
                                   xy_noise = 0.0003781199 / 2,
                                   env_noise = 0.01,
                                   device = "cuda",
                                   x_sample_fun = rnorm, 
                                   y_sample_fun = rnorm, ...) {
  
  with_no_grad({
  
    n <- xy$size()[1]
    
    t <- torch_rand(n, 1, device = device)
    
    z <- torch_tensor(cbind(x_sample_fun(n, ...),
                            y_sample_fun(n, ...)),
                      device = device)
    
    ## add some noise to xy to avoid overfitting to grid points
    target <- (xy$to(device = device) + (torch_rand_like(xy, device = device) * xy_noise)) - z
    
    coords <- z + t * target
    
  })
  
  ## also add some noise to environmental variables to help be robust to generative
  ## error from NichEncoder that could take environments off their normal manifold
  list(coords = coords, t = t, 
       env = env$to(device = device) + torch_randn_like(env, device = device) * env_noise, 
       target = target)
}

#test <- generate_training_data(batches[[1]]$xy, batches[[1]]$env)

norm_sampler <- function(sd) {
  function(n) {
    rnorm(n, sd = sd * 1.1)
  }
}
x_samp <- norm_sampler(sd(chelsa_df$x))
y_samp <- norm_sampler(sd(chelsa_df$y))
# train_dat <- generate_training_data(chelsa_df, 
#                                     x_sample_fun = x_samp,
#                                     y_sample_fun = y_samp)
# as_tibble(train_dat)

traj_net <- nn_module("TrajNet",
                      initialize = function(coord_dim, env_dim, breadths = c(512, 256, 128),
                                            t_encode = 32L, env_encode = 64L,
                                            device = "cuda") {
                        
                        if(length(breadths) != 3) {
                          stop("breadths should be length 3!")
                        }
                        
                        self$encode_t <- nn_linear(1L, t_encode)
                        
                        self$encode_env <- nn_linear(env_dim, env_encode)
                        
                        self$unet <- nndag(input = ~ coord_dim,
                                            t_encoded = ~ t_encode,
                                            env_encoded = ~ env_encode,
                                            e_1 = input + t_encoded + env_encoded ~ breadths[1],
                                            e_2 = e_1 + t_encoded + env_encoded ~ breadths[2],
                                            e_3 = e_2 + t_encoded + env_encoded ~ breadths[3],
                                            d_1 = e_3 + t_encoded + env_encoded ~ breadths[3],
                                            d_2 = d_1 + e_3 + t_encoded + env_encoded ~ breadths[2],
                                            d_3 = d_2 + e_2 + t_encoded + env_encoded ~ breadths[1],
                                            output = d_3 + e_1 + t_encoded + env_encoded ~ coord_dim,
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
                            y_new <- self$forward(coords = y$detach(), t = t$detach(), env = parms$env$detach())
                          })
                          list(as.vector(as.matrix(y_new$cpu())))
                        }
                        
                        self$sample_trajectory <- function(initial_vals,
                                                           env_vals,
                                                           steps = 200) {
                          parms <- list(env = env_vals$detach())
                          init_dim <- dim(initial_vals)
                          y <- as.vector(as.matrix(initial_vals$cpu()))
                          res <- lsoda(y,
                                       seq(0, 1, length.out = steps),
                                       self$desolve_wrapper, 
                                       parms)
                          times <- res[ , 1]
                          arr <- array(as.vector(res[ , -1]), 
                                       dim = c(steps, init_dim))
                          list(times = times, trajectories = arr)
                        }
                        
                      },
                      
                      forward = function(coords, t, env) {
                        
                        t_encoded <- self$encode_t(t)
                        env_encoded <- self$encode_env(env)
                        self$unet(input = coords, t_encoded = t_encoded, env_encoded = env_encoded)
                        
                      })

coord_dim <- 2
env_dim <- chelsa_df |> select(-x, -y, -batch) |> slice_head(n = 1) |> ncol()

checkpoint_fold <- "output/checkpoints/geo_env_model_v2_1"
if(!dir.exists(checkpoint_fold)) {
  dir.create(checkpoint_fold)
  trajnet <- traj_net(coord_dim, env_dim)
  trajnet_jit <- traj_net(coord_dim, env_dim)
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
  trajnet_jit <- torch_load(checkpoint)
  progress <- unglue_data(basename(checkpoint_files),
                          "epoch_{epoch}_batch_{batch_num}_model.pt")[most_recent, ]
  i <- 0
  start_epoch <- as.numeric(progress$epoch)
  epoch <- start_epoch
  batch_num <- 0
}
trajnet <- trajnet$cuda()
trajnet_jit <- trajnet_jit$cuda()
trajnet

test <- generate_training_data(batches[[1]]$xy,
                               batches[[1]]$env,
                               device = "cuda",
                               x_sample_fun = x_samp, 
                               y_sample_fun = y_samp)

trajnet_jit$encode_spec <- jit_trace(trajnet_jit$encode_env, test$env)
trajnet_jit$encode_t <- jit_trace(trajnet_jit$encode_t, test$t)
trajnet_jit$unet <- jit_trace(trajnet_jit$unet, test$coords, trajnet_jit$encode_t(test$t), 
                              trajnet_jit$encode_spec(test$env))

#trajnet$load_state_dict(trajnet_jit$state_dict())
#torch_save(trajnet, "test.pt")
#test_test <- torch_load("test.pt")

## test trajectory
# y_init <- torch_randn(1000, coord_dim)
# env <- torch_randn(1000, env_dim)
# traj_test <- trajnet$sample_trajectory(y_init$cuda(), env$cuda())
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
optimizer <- optim_adamw(trajnet_jit$parameters, lr = lr, weight_decay = 0.01)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = n_batches,
                          cycle_momentum = FALSE)

plot_result <- function(trajnet, env_batch, n_samps = 5000, steps = 500,
                        x_sample_fun = rnorm, y_sample_fun = rnorm, ...) {
  with_no_grad({
    dims <- env_batch$size()
    ind <- sample.int(dims[1], n_samps)
    env <- env_batch[ind, ]
    x_samp <- x_sample_fun(n_samps, ...)
    y_samp <- y_sample_fun(n_samps, ...)
    y_init <- torch_tensor(cbind(x_samp, y_samp))
    res <- trajnet$sample_trajectory(y_init$cuda(), env$cuda(), steps = steps)
    final <- res$trajectories[steps, , ]
    plot(chelsa_df |> select(x, y) |> slice_sample(n = n_samps), col = "green", pch = 19)
    points(final, col = "red", pch = 19)
    y_init$detach()
  })
  cuda_empty_cache()
}


losses <- numeric(n_epoch * n_batches)
i <- 0
#epoch <- 0
batch_num <- 0
total_iter <- n_epoch * n_batches
epoch_times <- numeric(total_iter)

cat("Plotting current result: \n")
ragg::agg_png(file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_plot.png")),
              width = 1000, height = 500)
try(plot_result(trajnet_jit, batches[[1]]$env, x_sample_fun = x_samp, y_sample_fun = y_samp))
dev.off()
trajnet$load_state_dict(trajnet_jit$state_dict())
torch_save(trajnet, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))

final_epoch <- start_epoch + n_epoch

for(epoch in start_epoch + seq_len(n_epoch)) {
  
  epoch_time <- Sys.time()
  
  batch_num <- 0
  for(b in batches) {
    batch_num <- batch_num + 1
    i <- i + 1
    
    optimizer$zero_grad()
    
    batch <- generate_training_data(batches[[batch_num]]$xy,
                                    batches[[batch_num]]$env,
                                    device = "cuda",
                                    x_sample_fun = x_samp, 
                                    y_sample_fun = y_samp)
    
    output <- trajnet_jit(batch$coords,
                      batch$t,
                      batch$env)
    
    loss <- trajnet_jit$loss_function(output, batch$target)
    losses[i] <- as.numeric(loss$cpu())
    
    if(batch_num %% 10 == 0) {
      
      cat("Epoch: ", epoch,
          "    batch: ", batch_num,
          "    iteration: ", i,
          "    loss: ", as.numeric(loss$cpu()),
          "\n")
      
    }
    
    if(i %% 1000 == 0) {
      cat("Plotting current result: \n")
      ragg::agg_png(file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_plot.png")),
                    width = 1000, height = 500)
      try(plot_result(trajnet_jit, batch$env$detach(), x_sample_fun = x_samp, y_sample_fun = y_samp))
      dev.off()
      trajnet$load_state_dict(trajnet_jit$state_dict())
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
              width = 1000, height = 500)
try(plot_result(trajnet_jit, batch$env, x_sample_fun = x_samp, y_sample_fun = y_samp))
dev.off()
trajnet$load_state_dict(trajnet_jit$state_dict())
torch_save(trajnet, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))
