library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(536567678)

c(env, s) %<-% read_rds("output/squamate_env_codes.rds")

env_dataset <- dataset(name = "env_ds",
                       initialize = function(env, spec) {
                         self$env <- torch_tensor(env)
                         self$spec <- torch_tensor(spec)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, ], spec = self$spec[i])
                       },
                       .length = function() {
                         self$env$size()[[1]]
                       })

train_ds <- env_dataset(env, s)
train_dl <- dataloader(train_ds, 900000, shuffle = TRUE)

env_vae_mod <- nn_module("ENV_VAE",
                         initialize = function(input_dim, spec_embed_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                           self$latent_dim <- latent_dim
                           self$input_dim <- input_dim
                           self$spec_embed_dim <- spec_embed_dim
                           self$encoder <- nndag(y = ~ input_dim,
                                                 #s = ~ spec_embed_dim,
                                                 e_1 = y ~ breadth,
                                                 e_2 = e_1 ~ breadth,
                                                 e_3 = e_2 ~ breadth,
                                                 means = e_3 ~ latent_dim,
                                                 logvars = e_3 ~ latent_dim,
                                                 .act = list(nn_relu,
                                                             logvars = nn_identity,
                                                             means = nn_identity))
                           
                           self$decoder <- nndag(z = ~ latent_dim,
                                                 #s = ~ spec_embed_dim,
                                                 d_1 = z ~ breadth,
                                                 d_2 = d_1 ~ breadth,
                                                 d_3 = d_2 ~ breadth,
                                                 out = d_3 ~ input_dim,
                                                 .act = list(nn_relu,
                                                             out = nn_identity))
                           
                           self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
                           
                         },
                         reparameterize = function(mean, logvar) {
                           std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                           eps <- torch_randn_like(std)
                           eps * std + mean
                         },
                         loss_function = function(reconstruction, input, mean, log_var,  
                                                  lambda = 1,
                                                  alpha = 1) {
                           
                           kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - self$latent_dim
                           #kl_spec <- ((1 - alpha) * torch_sum(torch_square(mean_spec), dim = 2L) + alpha * torch_sum(torch_abs(mean_spec), dim = 2L)) * lambda
                           recon1 <- torch_sum(torch_square(input - (reconstruction)), dim = 2L) / torch_exp(self$loggamma)
                           recon2 <- self$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                           recon_loss <- (torch_mean(recon1)) + torch_mean(recon2)
                           loss <- torch_mean(kl) + recon_loss
                           list(loss, torch_mean(recon1*torch_exp(self$loggamma)), torch_mean(kl))
                         },
                         encode = function(x) {
                           
                           self$encoder(x)
                         },
                         decode = function(z) {
                           self$decoder(z)
                         },
                         sample = function(n, s = NULL) {
                           z <- self$reparameterize(torch_zeros(n, self$latent_dim),
                                                    torch_zeros(n, self$latent_dim))
                           if(s$size[[1]] == 1 | s$size[[1]] == self$latent_dim) {
                             s <- s$`repeat`(c(n, 1))
                           }
                           self$decode(z, s)
                         },
                         forward = function(x, s) {
                           c(means, log_vars) %<-% self$encoder(y = x)
                           z <- self$reparameterize(means, log_vars)
                           list(self$decoder(z = z), x, means, log_vars)
                         }
                         
)

input_dim <- ncol(env)
spec_embed_dim <- 16L
latent_dim <- 16L
breadth <- 1024L

env_vae <- env_vae_mod(input_dim, spec_embed_dim, latent_dim, breadth, loggamma_init = 0)
env_vae <- env_vae$cuda()

num_epochs <- 1000

lr <- 0.002
optimizer <- optim_adamw(env_vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

old <- par(mfrow=c(4, 4))

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
    
    c(reconstruction, input, means, log_vars) %<-% env_vae(b$env$cuda())
    c(loss, reconstruction_loss, kl_loss) %<-% env_vae$loss_function(reconstruction, input, means, log_vars)
    
      cat("Epoch: ", epoch,
          "  batch: ", batchnum,
          "  loss: ", as.numeric(loss$cpu()),
          "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
          "  KL loss: ", as.numeric(kl_loss$cpu()),
          "  loggamma: ", as.numeric(env_vae$loggamma$cpu()),
          #"    loggamma: ", loggamma,
          "  cond. active dims: ", as.numeric((torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()),
          "\n")
      
    loss$backward()
    optimizer$step()
    scheduler$step()
  })
  
  samp_ind <- sample.int(means$size()[1], 5000)
  mean_samp <- as.matrix(means[samp_ind, ]$cpu())
  apply(mean_samp, 2, function(x) {qqnorm(x); qqline(x)})
  
  time <- Sys.time() - epoch_time
  epoch_times[i] <- time
  cat("Estimated time remaining: ")
  print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (num_epochs - epoch)))
  
  
}

options(torch.serialization_version = 2)
torch_save(env_vae, "data/env_vae_1_trained_fixed_16d_stage2_uncond.to")

par(old)