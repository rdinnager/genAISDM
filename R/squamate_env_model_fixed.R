library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(536567678)

squamate_train <- read_csv("output/squamate_training.csv")

species <- as.integer(as.numeric(as.factor(squamate_train$Binomial)))

squamate_train <- squamate_train |>
  select(-Binomial, -X, -Y, -n) |>
  as.matrix()
#squamate_test <- read_csv("output/squamate_testing.csv")

squamate_train <- scale(squamate_train)

scaling <- list(means = attr(squamate_train, "scaled:center"),
                sd = attr(squamate_train, "scaled:scale"))

write_rds(scaling, "output/squamate_env_scaling.rds")

na_mask <- apply(squamate_train, 2, function(x) as.numeric(!is.na(x)))

squamate_train[na_mask == 0] <- 0

env_dataset <- dataset(name = "env_ds",
                       initialize = function(env, mask, spec) {
                         self$env <- torch_tensor(env)
                         self$mask <- torch_tensor(mask)
                         self$spec <- torch_tensor(spec)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, ], mask = self$mask[i,], spec = self$spec[i])
                       },
                       .length = function() {
                         self$env$size()[[1]]
                       })

train_ds <- env_dataset(squamate_train, na_mask, species)
train_dl <- dataloader(train_ds, 900000, shuffle = TRUE)

#test <- train_dl$.iter()$.next()

env_vae_mod <- nn_module("ENV_VAE",
                         initialize = function(input_dim, n_spec, spec_embed_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                           self$latent_dim <- latent_dim
                           self$input_dim <- input_dim
                           self$n_spec <- n_spec
                           self$spec_embed_dim <- spec_embed_dim
                           self$encoder <- nndag(y = ~ input_dim,
                                                 s = ~ spec_embed_dim,
                                                 e_1 = y + s ~ breadth,
                                                 e_2 = e_1 + s ~ breadth,
                                                 e_3 = e_2 + s ~ breadth,
                                                 means = e_3 ~ latent_dim,
                                                 logvars = e_3 ~ latent_dim,
                                                 .act = list(nn_relu,
                                                             logvars = nn_identity,
                                                             means = nn_identity))
                           
                           self$decoder <- nndag(z = ~ latent_dim,
                                                 s = ~ spec_embed_dim,
                                                 d_1 = z + s ~ breadth,
                                                 d_2 = d_1 + s ~ breadth,
                                                 d_3 = d_2 + s ~ breadth,
                                                 out = d_3 ~ input_dim,
                                                 .act = list(nn_relu,
                                                             out = nn_identity))
                           
                           self$species_embedder_mean <- nn_embedding(n_spec, spec_embed_dim#,
                                                                      # .weight = torch_randn(n_spec,
                                                                      #                       spec_embed_dim) * 0.01
                                                                      )
                           # self$species_embedder_var <- nn_embedding(n_spec, spec_embed_dim,
                           #                                           .weight = -3 + torch_randn(n_spec,
                           #                                                                 spec_embed_dim) * 0.01
                           #                                           )
                           
                           self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
                           
                         },
                         reparameterize = function(mean, logvar) {
                           std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                           eps <- torch_randn_like(std)
                           eps * std + mean
                         },
                         loss_function = function(reconstruction, input, mask, mean, log_var, mean_spec, #log_var_spec, 
                                                  #loggamma,
                                                  lambda = 1,
                                                  alpha = 1) {
                           
                           kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - self$latent_dim
                           kl_spec <- ((1 - alpha) * torch_sum(torch_square(mean_spec), dim = 2L) + alpha * torch_sum(torch_abs(mean_spec), dim = 2L)) * lambda
                           recon1 <- torch_sum(torch_square(input - (reconstruction * mask)), dim = 2L) / torch_exp(self$loggamma)
                           recon2 <- self$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                           recon_loss <- (torch_sum(recon1) / torch_sum(mask)) + torch_mean(recon2)
                           loss <- torch_mean(kl + kl_spec) + recon_loss
                           list(loss, torch_mean(recon1*torch_exp(self$loggamma)), torch_mean(kl), torch_mean(kl_spec))
                         },
                         encode = function(x, s = NULL) {
                           if(is.null(s)) {
                             spec_embedding <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device)
                           } else {
                             if(s$size[[1]] == 1 | s$size[[1]] == self$n_spec) {
                               s <- s$`repeat`(s(x$size()[[1]], 1))
                             }
                             spec_embedding <- self$species_embedder_mean(s)
                           }
                           self$encoder(x, spec_embedding)
                         },
                         decode = function(z, s = NULL) {
                           if(is.null(s)) {
                             spec_embedding <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device)
                           } else {
                             if(s$size[[1]] == 1 | s$size[[1]] == self$n_spec) {
                               s <- s$`repeat`(c(x$size()[[1]], 1))
                             }
                             spec_embedding <- self$species_embedder_mean(s)
                           }
                           self$decoder(z, spec_embedding)
                         },
                         sample = function(n, s = NULL) {
                           z <- self$reparameterize(torch_zeros(n, self$latent_dim),
                                                    torch_zeros(n, self$latent_dim))
                           if(s$size[[1]] == 1 | s$size[[1]] == self$latent_dim) {
                             s <- s$`repeat`(c(n, 1))
                           }
                           self$decode(z, s)
                         },
                         forward = function(x, s = NULL) {
                           if(is.null(s)) {
                             spec_embedding_mean <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device)
                             #spec_embedding_log_var <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device) + 0.001
                           } else {
                             spec_embedding_mean <- self$species_embedder_mean(s)
                             #spec_embedding_log_var <- self$species_embedder_var(s)
                           }
                           #z_spec <- self$reparameterize(spec_embedding_mean, spec_embedding_log_var)
                           c(means, log_vars) %<-% self$encoder(y = x, s = spec_embedding_mean)
                           z <- self$reparameterize(means, log_vars)
                           list(self$decoder(z = z, s = spec_embedding_mean), x, spec_embedding_mean, means, log_vars)
                         }
                         
)

input_dim <- ncol(squamate_train)
n_spec <- n_distinct(species)
spec_embed_dim <- 32L
latent_dim <- 16L
breadth <- 1024L

env_vae <- env_vae_mod(input_dim, n_spec, spec_embed_dim, latent_dim, breadth, loggamma_init = -3)
env_vae <- env_vae$cuda()

num_epochs <- 3000

lr <- 0.002
optimizer <- optim_adamw(env_vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

epoch_times <- numeric(num_epochs * length(train_dl))
i <- 0
#b <- train_dl$.iter()$.next()

for (epoch in 1:num_epochs) {
  
  epoch_time <- Sys.time()
  batchnum <- 0
  coro::loop(for (b in train_dl) {
    
    batchnum <- batchnum + 1
    i <- i + 1
    optimizer$zero_grad()
    
    c(reconstruction, input, mean_spec, means, log_vars) %<-% env_vae(b$env$cuda(), b$spec$cuda())
    c(loss, reconstruction_loss, kl_loss, kl_loss_spec) %<-% env_vae$loss_function(reconstruction, input, b$mask$cuda(), means, log_vars, mean_spec, alpha = 0.9)
    
      cat("Epoch: ", epoch,
          "  batch: ", batchnum,
          "  loss: ", as.numeric(loss$cpu()),
          "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
          "  KL loss: ", as.numeric(kl_loss$cpu()),
          "  species KL loss: ", as.numeric(kl_loss_spec$cpu()),
          "  loggamma: ", as.numeric(env_vae$loggamma$cpu()),
          #"    loggamma: ", loggamma,
          "  cond. active dims: ", as.numeric((torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()),
          "  spec. cond. active dims: ", as.numeric(((torch_square(mean_spec)$mean(dim = 1L) / torch_square(mean_spec)$mean(dim = 1L)$sum()) > 0.01)$sum()$cpu()),
          "\n")
      
    loss$backward()
    optimizer$step()
    scheduler$step()
  })
  
  time <- Sys.time() - epoch_time
  epoch_times[i] <- time
  cat("Estimated time remaining: ")
  print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (num_epochs - epoch)))
  
}

options(torch.serialization_version = 2)
torch_save(env_vae, "data/env_vae_1_trained_fixed_32d.to")

