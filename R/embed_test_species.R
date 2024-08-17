library(tidyverse)
library(torch)
library(ape)
library(ggmulti)
library(deSolve)
library(zeallot)
library(cli)

options(torch.serialization_version = 2)
env_vae <- torch_load("data/env_vae_1_trained_fixed.to")
env_vae <- env_vae$cuda()

scaling <- read_rds("output/squamate_env_scaling.rds")

testing_df <- read_csv("output/squamate_testing.csv") 
testing_df <- testing_df |>
  filter(type == "between_species")

env_mat <- testing_df |>
  select(-Binomial, -X, -Y, -n, -type) |>
  as.matrix() |>
  scale(center = scaling$means, scale = scaling$sd)

species <- testing_df |>
  pull(Binomial) |>
  as.factor() |>
  as.integer()
specs <- species |>
  torch_tensor(device = "cuda")

na_mask <- apply(env_mat, 2, function(x) as.numeric(!is.na(x)))
mask <- torch_tensor(na_mask, device = "cuda")

env_mat[na_mask == 0] <- 0

env_dat <- env_mat |>
  torch_tensor(device = "cuda")

species_embedder <- nn_module("SpecEmbed",
                              initialize = function(env_vae, spec_embed_dim, n_spec, loggamma_init = -10) {
                                self$env_vae <- env_vae
                                self$spec_embed <- nn_embedding(n_spec, spec_embed_dim)
                                self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
                              },
                              loss_function = function(reconstruction, input, mask, mean, log_var) {
                                kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - self$env_vae$latent_dim
                                recon1 <- torch_sum(torch_square(input - (reconstruction * mask)), dim = 2L) / torch_exp(self$loggamma)
                                recon2 <- self$env_vae$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$env_vae$input_dim
                                recon3 <- (torch_sum(recon1) / torch_sum(mask))
                                recon_loss <- recon3 + torch_mean(recon2)
                                loss <- torch_mean(kl) + recon_loss
                                list(loss = loss, recon_loss = recon3 * torch_exp(self$loggamma))
                              },
                              forward = function(env, spec, mask) {
                                s <- self$spec_embed(spec)
                                c(means, log_vars) %<-% self$env_vae$encoder(y = env, s = s)
                                z <- self$env_vae$reparameterize(means, log_vars)
                                reconstruction <- self$env_vae$decoder(z = z, s = s)
                                c(loss, recon_loss) %<-% self$loss_function(reconstruction, env, mask, means, log_vars)
                                list(loss = loss, recon_loss = recon_loss, means = means, log_vars = log_vars)
                              })

s_embed <- species_embedder(env_vae, 16L, n_distinct(species))
s_embed <- s_embed$cuda()

## freeze env weights
walk(s_embed$env_vae$parameters,
     ~ .x$requires_grad_(FALSE))

n_epochs <- 5000

current_loss <- 9999

#s_embed <- torch_load("output/test_embedding_model.pt")
#s_embed <- s_embed$cuda()

#lr <- 0.002
lr <- 0.0002
optimizer <- optim_adam(s_embed$parameters, lr = lr)
# scheduler <- lr_one_cycle(optimizer, max_lr = lr,
#                           epochs = n_epochs, steps_per_epoch = 1,
#                           cycle_momentum = FALSE)

embedding_list <- list()

# for(epoch in cli_progress_along(1:n_epochs,
#                                 format = "Epoch: {epoch}; loss: {current_loss} {cli::pb_current}")) {
for(epoch in 1:n_epochs) {
  c(loss, recon_loss, means, log_vars) %<-% s_embed(env_dat, specs, mask)
  loss$backward()
  optimizer$step()
  #scheduler$step()
  current_loss <- as.numeric(loss$cpu()$detach())
  embedding_list[[epoch]] <- as.matrix(s_embed$spec_embed$weight$cpu())
  cat("Epoch: ", epoch, " Loss: ", current_loss, " Recon. Loss: ", as.numeric(recon_loss$cpu()$detach()), 
      "loggamma: ", as.numeric(s_embed$loggamma$cpu()$detach()), "\n")
}

write_rds(embedding_list, "output/test_embeddings_per_epoch_5000_2.rds")
options(torch.serialization_version = 2)
torch_save(s_embed, "output/test_embedding_model2.pt")

final_s <- s_embed$spec_embed(specs)
c(means, log_vars) %<-% s_embed$env_vae$encoder(y = env_dat, s = final_s)
final_z <- s_embed$env_vae$reparameterize(means, log_vars)
final_recon <- s_embed$env_vae$decoder(z = final_z, s = final_s)

recon_mat <- as.matrix(final_recon$cpu())
loss_mat <- (recon_mat - env_mat)^2
loss_mat <- loss_mat * na_mask
loss_means <- apply(loss_mat, 1, mean)
loss_df <- testing_df |>
  mutate(mean_loss = loss_means)

loss_summ <- loss_df |>
  group_by(Binomial) |>
  summarise(mean_loss = mean(mean_loss)) 

best <- loss_summ |>
  filter(mean_loss <= 0.2)

best_species <- testing_df |>
  select(Binomial) |>
  mutate(id = as.integer(as.factor(Binomial))) |>
  group_by(Binomial) |>
  summarise(Binomial = Binomial[1], id = id[1]) |>
  left_join(loss_summ)

write_csv(best_species, "output/best_embedded_species.csv")

