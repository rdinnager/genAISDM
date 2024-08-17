library(tidyverse)
library(torch)
library(deSolve)
library(dagnn)
options(torch.serialization_version = 2)

flow_1 <- torch_load("output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_2/epoch_6000_batch_10_model.pt")
flow_1 <- flow_1$cuda()

codes <- read_rds("output/squamate_env_latent_codes_for_stage2.rds") |>
  distinct(species, .keep_all = TRUE) |>
  select(starts_with("L")) |>
  slice_sample(prop = 1) |>
  as.matrix()

sampsize <- 2000
zs <- vector("list", nrow(codes))

total_iter <- nrow(codes)
epoch_times <- numeric(total_iter)

checkpoint_fold <- "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_2_distill_data/"

for(i in 1:nrow(codes)) {
  epoch_time <- Sys.time()
  with_no_grad({
    code_tens <- matrix(codes[i, ], ncol = ncol(codes), nrow = sampsize, byrow = TRUE) |>
      torch_tensor(device = "cuda")
    z0 <- torch_randn(sampsize, 3L, device = "cuda")
    z1 <- flow_1$sample_trajectory(initial_vals = z0$detach(), spec_vals = code_tens$detach(), steps = 250)
    zs[[i]] <- z1$trajectories[250, , ]
  })
  time <- Sys.time() - epoch_time
  epoch_times[i] <- time
  cat("Iteration: ", i, " Estimated time remaining: ")
  print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (total_iter - i)))
  if(i %% 500 == 0) {
    write_rds(zs, file.path(checkpoint_fold, glue::glue("zs_i_{i}.rds")))
  }
}