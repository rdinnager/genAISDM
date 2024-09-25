library(tidyverse)
library(torch)
library(deSolve)
library(dagnn)
options(torch.serialization_version = 2)

checkpoint_dir <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d"
files <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
file_info <- file.info(files)
latest <- which.max(file_info$mtime)
mod_file <- files[latest]
#flow_1 <- torch_load(mod_file)
#flow_1 <- flow_1$cuda()

dat <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  distinct(species, .keep_all = TRUE)

code_df <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds")

codes <- dat |>
  select(starts_with("L")) |>
  as.matrix()
  
specs <- dat |>
  pull(species)

sampsize <- 2000
zs <- vector("list", nrow(codes))

total_iter <- nrow(codes)
epoch_times <- numeric(total_iter)

checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d_distill_data/"

## trace module for speed
code_tens <- matrix(codes[1, ], ncol = ncol(codes), nrow = sampsize, byrow = TRUE) |>
  torch_tensor(device = "cuda")

z0 <- torch_randn(sampsize, 6L, device = "cuda")
t <- torch_tensor(matrix(seq(0, 1, length.out = sampsize), ncol = 1), device = "cuda")
#test <- flow_1(z0, t, code_tens)

#flow_1 <- torch_load(mod_file)
#flow_1 <- flow_1$cuda()

flow_1_jit <- torch_load(mod_file)
flow_1_jit <- flow_1_jit$cuda()

flow_1_jit$encode_spec <- jit_trace(flow_1_jit$encode_spec, code_tens)
flow_1_jit$encode_t <- jit_trace(flow_1_jit$encode_t, t)
flow_1_jit$unet <- jit_trace(flow_1_jit$unet, z0, flow_1_jit$encode_t(t), flow_1_jit$encode_spec(code_tens))

#flow_1_jit <- jit_trace(flow_1, z0, t, code_tens)

## test
# code_tens2 <- matrix(codes[101, ], ncol = ncol(codes), nrow = sampsize, byrow = TRUE) |>
#   torch_tensor(device = "cuda")
# 
# z02 <- torch_randn(sampsize, 6L, device = "cuda")
# t2 <- torch_tensor(matrix(seq(1, 0, length.out = sampsize), ncol = 1), device = "cuda")
# 
# test1 <- flow_1(z02, t2, code_tens2)
# test2 <- flow_1_jit(z02, t2, code_tens2)
# 
# torch_allclose(test1, test2)
# 
# ben <- bench::mark(flow_1(z02, t2, code_tens2),
#                    flow_1_jit(z02, t2, code_tens2),
#                    check = FALSE, filter_gc = FALSE, min_iterations = 1000)

## it is about twice as fast. Each iteration is actually about 5 times as fast
## but the jit one more frequently has slow iterations, which I assume are due to
## garbage collection though bench::mark does not detect them, it must be happening
## in C or CUDA?

for(i in 1:nrow(codes)) {
  
  file_name <- file.path(checkpoint_fold, glue::glue("zs_i_{i}.csv"))
  
  if(!file.exists(file_name)) {
    epoch_time <- Sys.time()
    with_no_grad({
      code_tens <- matrix(codes[i, ], ncol = ncol(codes), nrow = sampsize, byrow = TRUE) |>
        torch_tensor(device = "cuda")
      z0 <- torch_randn(sampsize, 6L, device = "cuda")
      z1 <- flow_1_jit$sample_trajectory(initial_vals = z0$detach(), spec_vals = code_tens$detach(), steps = 250)
      z <- z1$trajectories[250, , ]
    })
    z <- z |>
      as.data.frame() |>
      mutate(species = specs[i]) |>
      bind_cols(z0$cpu() |> as.matrix() |> as.data.frame() |> setNames(paste0("Z", 1:6L)))
    write_csv(z, file_name)
    time <- Sys.time() - epoch_time
    epoch_times[i] <- time
    cat("Iteration: ", i, " Estimated time remaining: ")
    print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (total_iter - i)))
  } else {
    time <- 0
    epoch_times[i] <- time
  }
    
}

# with_no_grad({
#   code_tens <- matrix(codes[i, ], ncol = ncol(codes), nrow = sampsize, byrow = TRUE) |>
#     torch_tensor(device = "cuda")
#   z0 <- torch_randn(sampsize, 6L, device = "cuda")
#   z1 <- flow_1$sample_trajectory(initial_vals = z0$detach(), spec_vals = code_tens$detach(), steps = 250)
#   z <- z1$trajectories[250, , ]
# })
# z2 <- z |>
#   as.data.frame() |>
#   mutate(species = specs[i])
# 
# z3 <- code_df |>
#   filter(species == specs[i]) 
# 
# plot(ze |> select(V3, V4), pch = 19, col = alpha("black", 0.1))
# #points(z2 |> select(V3, V4), col = "red")
# points(z3 |> select(V3, V4), pch = 19, col = alpha("green", 0.1))
library(data.table)
checkpoint_zs <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d_distill_data"
files <- list.files(checkpoint_zs, pattern = "csv", full.names = TRUE)
distill_data <- map(files, fread, .progress = TRUE)
distill_data <- list_rbind(distill_data)
fwrite(distill_data, "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_7d_distill_data/all_zs.csv")