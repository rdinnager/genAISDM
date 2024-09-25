library(tidyverse)
library(torch)
library(deSolve)
library(dagnn)
library(unglue)
library(data.table)
options(torch.serialization_version = 2)

distill_data <- fread("output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d_distill_data/all_zs.csv")

dat <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  distinct(species, .keep_all = TRUE)
spec_lats <- dat |>
  select(species, starts_with("L"))
distill_data <- distill_data |>
  left_join(spec_lats)

batch_size <- 5e5
n_epoch <- 6e3

distill_data <- distill_data |>
  sample_frac()

nbatches <- floor(nrow(distill_data) / batch_size) + 1
b <- gl(nbatches, batch_size, nrow(distill_data))

distill_data <- distill_data |>
  mutate(batch = b)

make_batch <- function(distill_data, batch_num = 1) {
  coord_dat <- distill_data |>
    filter(batch == batch_num)
  
  v <- coord_dat |>
    select(starts_with("V")) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  spec <- coord_dat |>
    select(starts_with("L")) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  z <- coord_dat |>
    select(starts_with("Z")) |>
    as.matrix() |>
    torch_tensor(pin_memory = TRUE)
  
  list(z = z, v = v, spec = spec)  
}

bs <- unique(distill_data$batch)

batches <- map(bs, ~ make_batch(distill_data, .x), .progress = TRUE)

generate_training_data <- function(z,
                                   v,
                                   spec,
                                   device = "cuda", ...) {
  
  with_no_grad({
    
    n <- v$size()[1]
    d <- v$size()[2]
    
    inds <- sample.int(n)
    z <- z[inds, ]$to(device = device)
    
    t <- torch_rand(n, 1, device = device)
    
    target <- v[inds, ]$to(device = device) - z
    
    coords <- z + t * target
    
  })
  
  list(coords = coords, t = t, spec = spec$to(device = device), target = target)
}

coord_dim <- 6
spec_dim <- distill_data |> select(starts_with("L")) |> slice_head(n = 1) |> ncol()

checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_distill_7d"

if(!dir.exists(checkpoint_fold)) {
  dir.create(checkpoint_fold)
  checkpoint_dir <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d"
  files <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  file_info <- file.info(files)
  latest <- which.max(file_info$mtime)
  mod_file <- files[latest]
  flow_2 <- torch_load(mod_file)
  i <- 0
  start_epoch <- 0
  epoch <- 0
  batch_num <- 0
} else {
  checkpoint_files <- list.files(checkpoint_fold, full.names = TRUE, pattern = ".pt")
  checkpoints <- file.info(checkpoint_files)
  most_recent <- which.max(checkpoints$mtime)
  checkpoint <- checkpoint_files[most_recent]
  flow_2 <- torch_load(checkpoint)
  progress <- unglue_data(basename(checkpoint_files),
                          "epoch_{epoch}_batch_{batch_num}_model.pt")[most_recent, ]
  i <- 0
  start_epoch <- as.numeric(progress$epoch)
  epoch <- start_epoch
  batch_num <- 0
}
flow_2 <- flow_2$cuda()
flow_2

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
lr <- 2e-5
optimizer <- optim_adamw(flow_2$parameters, lr = lr, weight_decay = 0.01)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = n_epoch, steps_per_epoch = n_batches,
                          cycle_momentum = TRUE, pct_start = 0.2)

plot_result <- function(flow_2, distill_data, n_specs = 9, steps = 250,
                        sample_fun = rnorm, ...) {
  with_no_grad({
    
    specsamp <- sample(unique(distill_data$species), n_specs)
    
    coord_dat <- distill_data |>
      filter(species %in% specsamp) |>
      slice_sample(n = 800)
    
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
    res <- flow_2$sample_trajectory(y_init$cuda(), spect$cuda(), steps = steps)
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
try(plot_result(flow_2, distill_data, steps = 2))
dev.off()
torch_save(flow_2, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))

final_epoch <- start_epoch + n_epoch

for(epoch in start_epoch + seq_len(n_epoch)) {
  
  epoch_time <- Sys.time()
  
  batch_num <- 0
  for(b in batches) {
    batch_num <- batch_num + 1
    i <- i + 1
    
    optimizer$zero_grad()
    
    batch <- generate_training_data(batches[[batch_num]]$z,
                                    batches[[batch_num]]$v,
                                    batches[[batch_num]]$spec,
                                    device = "cuda")
    
    output <- flow_2(batch$coords,
                      batch$t,
                      batch$spec)
    
    loss <- flow_2$loss_function(output, batch$target)
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
      try(plot_result(flow_2, distill_data, steps = 2))
      dev.off()
    }
    if(i %% 5000 == 0) {
      torch_save(flow_2, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))
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
try(plot_result(flow_2, distill_data, steps = 2))
dev.off()
torch_save(flow_2, file.path(checkpoint_fold, glue::glue("epoch_{epoch}_batch_{batch_num}_model.pt")))
