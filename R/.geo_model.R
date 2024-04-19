## Use a conditional diffusion model to model conditional probability of geographic
## coordinates with respect to the environment. Use consistency training to fit model

library(torch)
library(dagnn)
library(tidyverse)
library(terra)
library(sf)
library(concaveman)
library(zeallot)
library(ash)
library(ENMTools)
library(fields)
library(rnaturalearth)
library(terra)
library(conflicted)

conflict_prefer("extract", "terra")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

dat <- read_delim("data/full_dataset.csv", delim = ";")

pca_env <- rast("data/bioclim_pca_10.tif")

dat_env <- extract(pca_env, dat %>%
                     select(x = Longitude,
                            y = Latitude),
                   ID = FALSE)

full_dat <- dat %>%
  bind_cols(dat_env) %>%
  drop_na()

background <- full_dat %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326) %>%
  concaveman() %>%
  st_make_valid()

bg_env <- crop(pca_env, background) %>%
  mask(background)

bg_all <- as.data.frame(bg_env,
                        na.rm = TRUE,
                        xy = TRUE)

france <- ne_states(country = "France", returnclass = "sf")
bbox <- st_as_sf(as.polygons(bg_env, extent = TRUE))
france <- france %>%
  st_intersection(bbox) %>%
  st_union()

background <- france

bg_env <- crop(pca_env, background) %>%
  mask(background %>% st_as_sf())

bg_all <- as.data.frame(bg_env,
                        na.rm = TRUE,
                        xy = TRUE)

bbbox <- st_buffer(bbox, 1000000) %>%
  st_make_valid()

france_mask <- st_difference(bbbox, france)
plot(france_mask)

############# first train VAE in geographic space ##############

## setup model

sinusoid_pos_embed <- nn_module("SIN_POS_EMBED",
                                initialize = function(dim) {
                                  self$dim <- dim
                                },
                                forward = function(time) {

                                  device <- time$device
                                  half_dim <- self$dim %/% 2
                                  embeddings <- torch_log(10000) / (half_dim - 1)
                                  embeddings = torch_exp(torch_arange(0, half_dim - 1L, device = device) * -embeddings)
                                  embeddings = time * embeddings$unsqueeze(1L)
                                  embeddings = torch_cat(list(embeddings$sin(), embeddings$cos()), dim =-1L)
                                  embeddings

                                })

f_list <- map(1:29,
              ~ as.formula(paste0("c + d_", .x, " ~ 2L")))
names(f_list) <- paste0("d_", 2:30)
f_list <- c(list(y = ~ 2L, c = ~ 10L, d_1 = c + y ~ 2L), f_list)

decoder <- nndag(!!!f_list, .act = list(nn_tanh, d_30 = nn_identity))

xy_diff_mod <- nn_module("XY_DIFF",
                 initialize = function(input_dim, c_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                   self$latent_dim <- latent_dim
                   self$input_dim <- input_dim
                   self$c_dim <- c_dim
                   self$encoder <- nndag(y = ~ input_dim,
                                         c = ~ c_dim,
                                         #t = ~ ,
                                         #c_1_1 = c ~ breadth,
                                         #c_1_2 = c_1_1 ~ breadth,
                                         #c_2_1 = c ~ breadth,
                                         #c_2_2 = c_2_1 ~ breadth,
                                         #c_3_1 = c ~ breadth,
                                         #c_3_2 = c_3_1 ~ breadth,
                                         e_1 = c + y ~ breadth,
                                         e_2 = c + e_1 ~ breadth,
                                         e_3 = c + e_2 ~ breadth,
                                         e_4 = c + e_3 ~ breadth,
                                         out = e_4 ~ input_dim,
                                         logvars = e_4 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     logvars = nn_identity,
                                                     means = nn_identity))

                   self$decoder <- nndag(z = ~ latent_dim,
                                         #c = ~ c_dim,
                                         #c_1_1 = c ~ breadth,
                                         #c_1_2 = c_1_1 ~ breadth,
                                         #c_2_1 = c ~ breadth,
                                         #c_2_2 = c_2_1 ~ breadth,
                                         #c_3_1 = c ~ breadth,
                                         #c_3_2 = c_3_1 ~ breadth,
                                         d_1 = z ~ breadth,
                                         d_2 = d_1 ~ breadth,
                                         d_3 = d_2 ~ breadth,
                                         d_4 = d_3 ~ breadth,
                                         out = d_4 ~ input_dim,
                                         .act = list(nn_relu,
                                                     out = nn_identity))

                   self$loggamma <- nn_parameter(torch_tensor(loggamma_init))

                 },
                 reparameterize = function(mean, logvar) {
                   std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                   eps <- torch_randn_like(std)
                   eps * std + mean
                 },
                 loss_function = function(reconstruction, input, mean, log_var) {

                   kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - latent_dim
                   recon1 <- torch_sum(torch_square(input - reconstruction), dim = 2L) / torch_exp(self$loggamma)
                   recon2 <- self$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                   loss <- torch_mean(recon1 + recon2 + kl)
                   list(loss, torch_mean(recon1*torch_exp(self$loggamma)), torch_mean(kl))
                 },
                 encode = function(x, c = NULL) {
                   self$encoder(c)
                 },
                 decode = function(z, c = NULL) {
                   self$decoder(z)
                 },
                 forward = function(x, c = NULL) {
                   c(means, log_vars) %<-% self$encoder(c)
                   z <- self$reparameterize(means, log_vars)
                   list(self$decoder(z), x, means, log_vars)
                 })

################### run XY model ########################
## dataloader
## Note: Should add uniform noise to x and y coordinates to account
## grid resolution, any xy value in the grid cell validly has those
## corresponding env variables

xy_dataset <- dataset(name = "xy_ds",
                           initialize = function(dat, res) {
                             self$xy <- torch_tensor(dat[ , 1:2])
                             self$env <- torch_tensor(dat[ , -1:-2])
                             self$res <- res
                           },
                           .getbatch = function(i) {
                             xy <- self$xy[i, ]
                             list(xy = xy + ((torch_rand_like(xy) - 0.5) * self$res),
                                  env = self$env[i, ])
                           },
                           .length = function() {
                             self$xy$size()[[1]]
                           })
train_ds <- xy_dataset(as.matrix(bg_all), res(bg_env))
train_dl <- dataloader(train_ds, 10000, shuffle = TRUE)

input_dim <- 2L
c_dim <- 10L
latent_dim <- 64L
breadth <- 1024L

xy_vae <- xy_vae_mod(input_dim, c_dim, latent_dim, breadth, loggamma_init = -3)
xy_vae <- xy_vae$cuda()

num_epochs <- 25000

lr <- 0.002
optimizer <- optim_adam(xy_vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

for (epoch in 1:num_epochs) {

    batchnum <- 0
    coro::loop(for (b in train_dl) {

        batchnum <- batchnum + 1
        optimizer$zero_grad()

        c(reconstruction, input, mean, log_var) %<-% xy_vae(b$xy$cuda(), b$env$cuda())
        c(loss, reconstruction_loss, kl_loss) %<-% xy_vae$loss_function(reconstruction, input, mean, log_var)

        if(batchnum %% 2 == 0) {

            cat("Epoch: ", epoch,
                "  batch: ", batchnum,
                "  loss: ", as.numeric(loss$cpu()),
                "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "  KL loss: ", as.numeric(kl_loss$cpu()),
                "  loggamma: ", as.numeric(xy_vae$loggamma$cpu()),
                #"    loggamma: ", loggamma,
                "  cond. active dims: ", as.numeric((torch_exp(log_var)$mean(dim = 1L) < 0.5)$sum()$cpu()),
                "\n")

        }
        loss$backward()
        optimizer$step()
        scheduler$step()
    })
}

torch_save(xy_vae, "data/xy_vae_1.to")
xy_vae <- torch_load("data/xy_vae_1.to")
xy_vae <- xy_vae$cuda()
