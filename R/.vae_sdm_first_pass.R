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

bbbox <- st_buffer(bbox, 1000000) %>%
  st_make_valid()

france_mask <- st_difference(bbbox, france)
plot(france_mask)

############# first train VAE in geographic space ##############

## setup model


xy_vae_mod <- nn_module("XY_VAE",
                 initialize = function(input_dim, c_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                   self$latent_dim <- latent_dim
                   self$input_dim <- input_dim
                   self$c_dim <- c_dim
                   self$encoder <- nndag(#y = ~ input_dim,
                                         c = ~ c_dim,
                                         #c_1_1 = c ~ breadth,
                                         #c_1_2 = c_1_1 ~ breadth,
                                         #c_2_1 = c ~ breadth,
                                         #c_2_2 = c_2_1 ~ breadth,
                                         #c_3_1 = c ~ breadth,
                                         #c_3_2 = c_3_1 ~ breadth,
                                         e_1 = c ~ breadth,
                                         e_2 = e_1 ~ breadth,
                                         e_3 = e_2 ~ breadth,
                                         e_4 = e_3 ~ breadth,
                                         means = e_4 ~ latent_dim,
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

################### now train VAE in environmental space ####################
## setup model

env_vae_mod <- nn_module("ENV_VAE",
                 initialize = function(input_dim, c_dim, c_embed_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                   self$latent_dim <- latent_dim
                   self$input_dim <- input_dim
                   self$c_dim <- c_dim
                   self$c_embed_dim <- c_embed_dim
                   self$encoder <- nndag(y = ~ input_dim,
                                         c = ~ c_embed_dim,
                                         e_1 = y + c ~ breadth,
                                         e_2 = e_1 + c ~ breadth,
                                         e_3 = e_2 + c ~ breadth,
                                         means = e_3 ~ latent_dim,
                                         logvars = e_3 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     logvars = nn_identity,
                                                     means = nn_identity))

                   self$encoder2 <- nndag(y2 = ~ latent_dim,
                                         c2 = ~ c_embed_dim,
                                         e2_1 = y2 + c2 ~ breadth,
                                         e2_2 = e2_1 + c2 ~ breadth,
                                         e2_3 = e2_2 + c2 ~ breadth,
                                         means = e2_3 ~ latent_dim,
                                         logvars = e2_3 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     logvars = nn_identity,
                                                     means = nn_identity))

                   self$decoder <- nndag(z = ~ latent_dim,
                                         c = ~ c_embed_dim,
                                         d_1 = z + c ~ breadth,
                                         d_2 = d_1 + c ~ breadth,
                                         d_3 = d_2 + c ~ breadth,
                                         out = d_3 ~ input_dim,
                                         .act = list(nn_relu,
                                                     out = nn_identity))

                   self$decoder2 <- nndag(z2 = ~ latent_dim,
                                         c2 = ~ c_embed_dim,
                                         d2_1 = z2 + c2 ~ breadth,
                                         d2_2 = d2_1 + c2 ~ breadth,
                                         d2_3 = d2_2 + c2 ~ breadth,
                                         out = d2_3 ~ latent_dim,
                                         .act = list(nn_relu,
                                                     out = nn_identity))

                   self$embedder <- nn_embedding(c_dim, c_embed_dim)
                   self$embedder2 <- nn_embedding(c_dim, c_embed_dim)

                   self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
                   self$loggamma2 <- nn_parameter(torch_tensor(loggamma_init))

                 },
                 reparameterize = function(mean, logvar) {
                   std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                   eps <- torch_randn_like(std)
                   eps * std + mean
                 },
                 loss_function = function(reconstruction, input, mean, log_var, embedding, loggamma,
                                          lambda = 1) {
                   ridge <- torch_sum(embedding^2, dim = 2L) * lambda
                   kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - latent_dim
                   recon1 <- torch_sum(torch_square(input - reconstruction), dim = 2L) / torch_exp(loggamma)
                   recon2 <- self$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                   loss <- torch_mean(recon1 + recon2 + kl + ridge)
                   list(loss, torch_mean(recon1*torch_exp(self$loggamma)), torch_mean(kl), torch_mean(ridge))
                 },
                 encode = function(x, c = NULL) {
                   if(is.null(c)) {
                     c_embedding <- torch_zeros(x$size()[[1]], self$c_embed_dim, device = x$device)
                   } else {
                     if(c$size[[1]] == 1 | c$size[[1]] == self$c_dim) {
                       c <- c$`repeat`(c(x$size()[[1]], 1))
                     }
                     c_embedding <- self$embedder(c)
                   }
                   self$encoder(x, c_embedding)
                 },
                 decode = function(z, c = NULL) {
                   if(is.null(c)) {
                     c_embedding <- torch_zeros(x$size()[[1]], self$c_embed_dim, device = x$device)
                   } else {
                     if(c$size[[1]] == 1 | c$size[[1]] == self$c_dim) {
                       c <- c$`repeat`(c(x$size()[[1]], 1))
                     }
                     c_embedding <- self$embedder(c)
                   }
                   self$decoder(z, c_embedding)
                 },
                 sample = function(n, c = NULL) {
                   z <- self$reparameterize(torch_zeros(n, self$latent_dim),
                                            torch_zeros(n, self$latent_dim))
                   if(c$size[[1]] == 1 | c$size[[1]] == self$latent_dim) {
                     c <- c$`repeat`(c(n, 1))
                   }
                   self$decode(z, c)
                 },
                 forward = function(x, c = NULL) {
                   if(is.null(c)) {
                     c_embedding <- torch_zeros(x$size()[[1]], self$c_embed_dim, device = x$device)
                   } else {
                     c_embedding <- self$embedder(c)
                   }
                   c(means, log_vars) %<-% self$encoder(x, c_embedding)
                   z <- self$reparameterize(means, log_vars)
                   list(self$decoder(z, c_embedding), x, c_embedding, means, log_vars)
                 },
                 forward2 = function(x, c = NULL) {
                   if(is.null(c)) {
                     c_embedding <- torch_zeros(x$size()[[1]], self$c_embed_dim, device = x$device)
                   } else {
                     c_embedding <- self$embedder2(c)
                   }
                   c(means, log_vars) %<-% self$encoder2(x, c_embedding)
                   z <- self$reparameterize(means, log_vars)
                   list(self$decoder2(z, c_embedding), x, c_embedding, means, log_vars)
                 }
                 )

## split out a test set to look at later
## we will remove some entire species, and also some individual
## occurrence points for other species
set.seed(1234)
whole_spec <- sample(unique(full_dat$Label), 350)
partial_spec <- sample(setdiff(unique(full_dat$Label), whole_spec), 500)

full_dat <- full_dat %>%
  mutate(training = 1) %>%
  mutate(training = ifelse(Label %in% whole_spec, 0, training)) %>%
  mutate(training = ifelse(Label %in% partial_spec, rbinom(n(), 1, 0.75), training)) %>%
  mutate(Label = as.factor(Label))

training_dat <- full_dat %>%
  filter(training == 1) %>%
  select(Label, PC1:PC10)

test_dat <- full_dat %>%
  filter(training == 0) %>%
  select(Label, PC1:PC10) %>%
  mutate(type = ifelse(Label %in% whole_spec, "whole", "partial"))

## dataloader
env_dataset <- dataset(name = "env_ds",
                           initialize = function(dat, fac) {
                             self$env <- torch_tensor(dat)
                             self$spec <- torch_tensor(fac)
                           },
                           .getbatch = function(i) {
                             list(env = self$env[i, ], spec = self$spec[i])
                           },
                           .length = function() {
                             self$env$size()[[1]]
                           })
train_ds <- env_dataset(as.matrix(training_dat[ , -1]), training_dat %>% pull(Label))
train_dl <- dataloader(train_ds, 20000, shuffle = TRUE)

input_dim <- 10L
c_dim <- nlevels(training_dat$Label)
c_embed_dim <- 64L
latent_dim <- 64L
breadth <- 1024L

env_vae <- env_vae_mod(input_dim, c_dim, c_embed_dim, latent_dim, breadth, loggamma_init = -3)
env_vae <- env_vae$cuda()

num_epochs <- 25000

lr <- 0.002
optimizer <- optim_adam(env_vae$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

for (epoch in 1:num_epochs) {

    batchnum <- 0
    coro::loop(for (b in train_dl) {

        batchnum <- batchnum + 1
        optimizer$zero_grad()

        c(reconstruction, input, embedding, mean, log_var) %<-% env_vae(b$env$cuda(), b$spec$cuda())
        c(loss, reconstruction_loss, kl_loss, ridge_loss) %<-% env_vae$loss_function(reconstruction, input, mean, log_var, embedding, env_vae$loggamma)

        if(batchnum %% 4 == 0) {

            cat("Epoch: ", epoch,
                "  batch: ", batchnum,
                "  loss: ", as.numeric(loss$cpu()),
                "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "  KL loss: ", as.numeric(kl_loss$cpu()),
                "  loggamma: ", as.numeric(env_vae$loggamma$cpu()),
                "  ridge loss: ", as.numeric(ridge_loss$cpu()),
                #"    loggamma: ", loggamma,
                "  cond. active dims: ", as.numeric((torch_exp(log_var)$mean(dim = 1L) < 0.5)$sum()$cpu()),
                "\n")

        }
        loss$backward()
        optimizer$step()
        scheduler$step()
    })
}

torch_save(env_vae, "data/env_vae_1_trained_1.to")
env_vae <- torch_load("data/env_vae_1_trained_1.to")

#################### Stage 2 VAE ###################

train_dl2 <- dataloader(train_ds, 20000, shuffle = FALSE)
dat <- list()
i <- 0
coro::loop(for (b in train_dl2) {
  i <- i + 1
  with_no_grad({
    c(reconstruction, input, embedding, mean, log_var) %<-% env_vae(b$env$cuda(), b$spec$cuda())
  })
  dat[[i]] <- list(means = as.data.frame(as.matrix(mean$cpu())),
                   logvars = as.data.frame(as.matrix(log_var$cpu())))
  print(i)
})

mean_df <- dat %>%
  map("means") %>%
  list_rbind()

logvar_df <- dat %>%
  map("logvars") %>%
  list_rbind()

logvar_mean <- map_dbl(logvar_df, ~ mean(exp(.x)))
hist(logvar_mean, breaks = 50)
hist(exp(logvar_df[ , "V30"]))


## dataloader
env_dataset2 <- dataset(name = "env_ds2",
                           initialize = function(m, s, fac) {
                             self$m <- torch_tensor(m)
                             self$s <- torch_tensor(s)
                             self$spec <- torch_tensor(fac)
                           },
                           .getbatch = function(i) {
                             m <- self$m[i, ]
                             s <- self$s[i, ]
                             z <- (torch_randn_like(m) * torch_exp(s)) + m
                             list(z = z, spec = self$spec[i])
                           },
                           .length = function() {
                             self$m$size()[[1]]
                           })
train2_ds2 <- env_dataset2(as.matrix(mean_df), as.matrix(logvar_df),
                           training_dat %>% pull(Label))
train2_dl2 <- dataloader(train2_ds2, 20000, shuffle = TRUE)

num_epochs <- 25000

lr <- 0.002
optimizer2 <- optim_adam(env_vae$parameters, lr = lr)
scheduler2 <- lr_one_cycle(optimizer2, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train2_dl2),
                          cycle_momentum = FALSE)

for (epoch in 1:num_epochs) {

    batchnum <- 0
    coro::loop(for (b in train2_dl2) {

        batchnum <- batchnum + 1
        optimizer$zero_grad()

        c(reconstruction, input, embedding, mean, log_var) %<-% env_vae$forward2(b$z$cuda(), b$spec$cuda())
        c(loss, reconstruction_loss, kl_loss, ridge_loss) %<-% env_vae$loss_function(reconstruction, input, mean, log_var, embedding, env_vae$loggamma2)

        if(batchnum %% 4 == 0) {

            cat("Epoch: ", epoch,
                "  batch: ", batchnum,
                "  loss: ", as.numeric(loss$cpu()),
                "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "  KL loss: ", as.numeric(kl_loss$cpu()),
                "  loggamma: ", as.numeric(env_vae$loggamma2$cpu()),
                "  ridge loss: ", as.numeric(ridge_loss$cpu()),
                #"    loggamma: ", loggamma,
                "  cond. active dims: ", as.numeric((torch_exp(log_var)$mean(dim = 1L) < 0.5)$sum()$cpu()),
                "\n")

        }
        loss$backward()
        optimizer2$step()
        scheduler2$step()
    })
}

torch_save(env_vae, "data/env_vae_1_trained_2.to")
env_vae <- torch_load("data/env_vae_1_trained_2.to")
env_vae <- env_vae$cuda()

################### evaluate model #####################

# test_dataset <- dataset(name = "test_ds",
#                            initialize = function(dat, fac) {
#                              self$env <- torch_tensor(dat)
#                              self$spec <- torch_tensor(fac)
#                            },
#                            .getbatch = function(i) {
#                              list(env = self$env[i, ], spec = self$spec[i])
#                            },
#                            .length = function() {
#                              self$env$size()[[1]]
#                            })
#
# test_ds <- env_dataset(as.matrix(test_dat[ , -1]), test_dat %>% pull(Label))
# test_dl <- dataloader(test_ds, 20000, shuffle = TRUE)

get_spec_embedding <- function(n = 10000, species = NULL, embedding = NULL) {
  if(is.null(species)) {
    if(is.null(embedding[[1]])) {
      stop("One of species or embedding must be non-NULL")
    } else {
      if(embedding[[1]]$size()[[1]] == 1 | embedding[[1]]$size()[[1]] == env_vae$c_embed_dim) {
        c1 <- embedding[[1]]$`repeat`(c(n, 1))
      } else {
        c1 <- embedding[[1]]
      }
      if(length(embedding) > 1) {
        if(embedding[[2]]$size()[[1]] == 1 | embedding[[2]]$size()[[1]] == env_vae$c_embed_dim) {
          c2 <- embedding[[2]]$`repeat`(c(n, 1))
        } else {
          c2 <- embedding[[2]]
        }
      } else {
        c2 <- c1
      }
    }
  } else {
    if(species$size()[[1]] == 1) {
      c <- species$`repeat`(n)$cuda()
    } else {
      c <- species$cuda()
    }
    c1 <- env_vae$embedder2(c)
    c2 <- env_vae$embedder(c)
  }
  list(c1, c2)
}

sample_env <- function(n = 10000, species = NULL, embedding = NULL, env_vae) {

  with_no_grad({
    c(c1, c2) %<-% get_spec_embedding(n, species, embedding)
    z <- env_vae$reparameterize(torch_zeros(n, env_vae$latent_dim)$cuda(),
                                torch_zeros(n, env_vae$latent_dim)$cuda())
    # z2 <- env_vae$decoder2(z, c1)
    # z2 <- env_vae$decoder2(z, torch_zeros_like(z)$cuda())
    env_samp <- env_vae$decoder(z, c2)
  })
  tt <- as.matrix(env_samp$cpu())

}

evaluate_species <- function(species = NULL, embedding = NULL, env_vae, xy_vae,
                             presence = NULL, pres_xy = NULL, test_xy = NULL,
                             background = NULL,
                             test = NULL,
                             do_plots = TRUE,
                             do_rf = TRUE,
                             trans = TRUE,
                             val_lims = NULL) {



  if(!is.null(species)) {
    species_t <- torch_tensor(species)$cuda()
  } else {
    species_t <- NULL
  }
  env_samp <- map(1:10, ~ sample_env(species = species_t, embedding = embedding, env_vae = env_vae))
  species2 <- get_spec_embedding(species = species_t, embedding = embedding)
  # test <- xy_vae(torch_zeros(nrow(env_samp[[1]]), 2),
  #                torch_tensor(env_samp[[1]])$cuda())[[1]]$cpu()
  geo_samp <- map(env_samp,
                  ~ xy_vae(torch_zeros(nrow(.x), 2),
                           torch_tensor(.x)$cuda())[[1]]$cpu() %>%
                   as.matrix())

  env_samp <- env_samp %>%
    do.call(rbind, .)
  geo_samp <- geo_samp %>%
    do.call(rbind, .)

  if(do_plots) {
    p <- ggplot(as.data.frame(env_samp), aes(V1, V2)) +
      geom_density2d_filled(bins = 30)

    if(!is.null(presence)) {
      p <- p +
        geom_point(aes(PC1, PC2), data = presence, col = "red")
    }
    if(!is.null(test)) {
      p <- p +
        geom_point(aes(PC1, PC2), data = test, col = "orange")
    }

    p <- p + theme_minimal() +
      theme(legend.position = "none")

    p
  } else {
    p <- NULL
  }

  bins <- ash::bin2(geo_samp, nbin = c(300, 300))
  ashs <- ash::ash2(bins)

  ap <- fields::interp.surface(list(x = ashs$x, y = ashs$y,
                                      z = ashs$z),
                                 background)

  if(!is.null(test_xy)) {
    pp <- fields::interp.surface(list(x = ashs$x, y = ashs$y,
                                      z = ashs$z),
                                 as.matrix(test_xy[ , 2:3]))

    p_df <- data.frame(p = c(pp, ap),
                       truth = as.factor(c(rep(1, length(pp)),
                                 rep(0, length(ap)))))
    auc_test <- yardstick::roc_auc(p_df, p, truth = truth, event_level = "second")
  } else {
    auc_test <- NULL
  }

  if(!is.null(pres_xy)) {

    pp <- fields::interp.surface(list(x = ashs$x, y = ashs$y,
                                      z = ashs$z),
                                 as.matrix(pres_xy[ , 2:3]))

    p_df <- data.frame(p = c(pp, ap),
                       truth = as.factor(c(rep(1, length(pp)),
                                 rep(0, length(ap)))))
    auc_train <- yardstick::roc_auc(p_df, p, truth = truth, event_level = "second")

    espec <- ENMTools::enmtools.species(
      range = bg_env,
      presence.points = terra::vect(pres_xy[ , -1], geom = c("x", "y")),
    )

    if(do_rf) {
      rf_comp <- try(ENMTools::enmtools.rf(espec,
                                       bg_env,
                                       eval = TRUE))
    } else {
      rf_comp <- NULL
    }
    if(inherits(rf_comp, "try-error")) {
      rf_comp <- NULL
    }

    if(!is.null(test_xy) & !is.null(rf_comp)) {
      auc2 <- dismo::evaluate(as.matrix(test_xy[, -1]), background, rf_comp$model,
                          bg_env)
    } else {
      auc2 <- NULL
    }

  } else {
    auc_train <- NULL
    auc2 <- NULL
    rf_comp <- NULL
  }

  ash_df <- ashs$z %>%
    as.data.frame() %>%
    mutate(x_num = 1:n()) %>%
    pivot_longer(cols = -x_num)
  x_df <- data.frame(x = ashs$x, x_num = 1:length(ashs$x))
  y_df <- data.frame(y = ashs$y, name = paste0("V", 1:length(ashs$y)))

  ash_df <- ash_df %>%
    left_join(x_df) %>%
    left_join(y_df)

  if(do_plots) {

    p2 <- ggplot(ash_df, aes(x, y)) +
      geom_raster(aes(fill = value)) +
      #geom_contour(aes(z = value), color = "white") +
      geom_sf(data = france_mask, fill = "white", inherit.aes = FALSE) +
      coord_sf(xlim = c(-5.2, 9.6), ylim = c(41.2, 51.1)) +
      ylab("") + xlab("")

    if(trans) {
      if(!is.null(val_lims)) {
        p2 <- p2 + scale_fill_viridis_c(option = "inferno", trans = "sqrt", limits = val_lims)
      } else {
        p2 <- p2 + scale_fill_viridis_c(option = "inferno", trans = "sqrt")
      }
    } else {
      if(!is.null(val_lims)) {
        p2 <- p2 + scale_fill_viridis_c(option = "inferno")
      } else {
        p2 <- p2 + scale_fill_viridis_c(option = "inferno", limits = val_lims)
      }

    }

    if(!is.null(pres_xy)) {
      p2 <- p2 +
        geom_point(aes(x, y), data = pres_xy, fill = "red", colour = "black", shape = 21, alpha = 0.7)
    }
    if(!is.null(test_xy)) {
      p2 <- p2 +
        geom_point(aes(x, y), data = test_xy, fill = "orange", colour = "black", shape = 21, alpha = 0.7)
    }

    p2 <- p2 +
      theme_minimal()

    p2
  } else {
    p2 <- NULL
  }

  list(env_plot = p, geo_plot = p2, test_auc = auc_test, rf_auc = auc2,
       train_auc = auc_train, rf_mod = rf_comp)

}

set.seed(154)

specieses <- unique(test_dat %>%
                      filter(type == "partial") %>%
                      pull(Label))
species <- sample(specieses, 1)

presence <- training_dat %>%
  filter(Label == species)

test_pres <- test_dat %>%
  filter(Label == species)

pres_xy <- full_dat %>%
  filter(training == 1) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  filter(Label == species)

test_xy <- full_dat %>%
  filter(training == 0) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  filter(Label == species)

bg_ds <- xy_dataset(as.matrix(bg_all), res(bg_env))
bg_dl <- dataloader(bg_ds, 10000, shuffle = TRUE)

background <- dataloader_make_iter(bg_dl) %>%
  dataloader_next() %>%
  .$xy %>%
  as.matrix()

test_eval <- evaluate_species(species, env_vae = env_vae, xy_vae = xy_vae,
                              presence = presence, pres_xy = pres_xy,
                              test_xy = test_xy, background = background,
                              test = test_pres,
                              do_plots = FALSE)

pres_list <- training_dat %>%
  group_by(Label) %>%
  group_nest(keep = TRUE)
pres_xy_list <- full_dat %>%
  filter(training == 1) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  group_by(Label) %>%
  group_nest(keep = TRUE)
test_pres_list <- test_dat %>%
  filter(type == "partial") %>%
  group_by(Label) %>%
  group_nest(keep = TRUE)
test_xy_list <- full_dat %>%
  filter(training == 0) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  group_by(Label) %>%
  group_nest(keep = TRUE)

for_testing_dat <- test_pres_list %>%
  rename(test_env_data = data) %>%
  left_join(pres_list %>% rename(train_env_data = data)) %>%
  left_join(pres_xy_list %>% rename(train_xy_data = data)) %>%
  left_join(test_xy_list %>% rename(test_xy_data = data))

for_testing_dat <- for_testing_dat %>%
  rowwise() %>%
  mutate(n_train_xy = list(nrow(train_xy_data)),
         n_train_env = list(nrow(train_env_data)),
         n_test_xy = list(nrow(test_xy_data)),
         n_test_env = list(nrow(test_env_data))) %>%
  ungroup() %>%
  filter(!sapply(n_train_xy, is.null))

all_auc <- pmap(for_testing_dat,
                ~ suppressMessages(evaluate_species(..1,
                                   env_vae = env_vae,
                                   xy_vae = xy_vae,
                                   presence = ..3,
                                   pres_xy = ..4,
                                   test_xy = ..5,
                                   background = background,
                                   test = ..2,
                                   do_plots = FALSE)),
                .progress = TRUE)

write_rds(all_auc, "data/all_tests_aucs.rds")

test_aucs <- map(all_auc, ~ .x$test_auc$.estimate[1]) %>%
  compact() %>%
  unlist()

rf_aucs <- map_if(all_auc, ~ !is.null(.x$rf_auc),
                  ~ .x$rf_auc@auc,
                  .else = ~ NULL) %>%
  compact() %>%
  unlist()

quantile(test_aucs, c(0.025, 0.5, 0.975), na.rm = TRUE)
quantile(rf_aucs, c(0.025, 0.5, 0.975), na.rm = TRUE)
mean(test_aucs, na.rm = TRUE)
mean(rf_aucs, na.rm = TRUE)

############## find unseen species ###########
unseen_dat <- test_dat %>%
  filter(type == "whole") %>%
  mutate(Label = droplevels(Label))

## dataloader
unseen_dataset <- dataset(name = "unseen_ds",
                           initialize = function(dat, fac) {
                             self$env <- torch_tensor(dat)
                             self$spec <- torch_tensor(fac)
                           },
                           .getbatch = function(i) {
                             list(env = self$env[i, ], spec = self$spec[i])
                           },
                           .length = function() {
                             self$env$size()[[1]]
                           })
unseen_ds <- env_dataset(as.matrix(unseen_dat[ , 2:11]), unseen_dat %>% pull(Label))
unseen_dl <- dataloader(unseen_ds, 20000, shuffle = TRUE)


## species search module
species_search_mod <- nn_module(
  initialize = function(n, env_vae, loggamma_init = -3) {
    self$env_vae <- env_vae$clone()
    walk(self$env_vae$parameters, ~ .x$requires_grad_(FALSE))
    self$embedder <- nn_embedding(n, env_vae$c_embed_dim,
                                  .weight = torch_zeros(n, env_vae$c_embed_dim))
    self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
  },
  loss_function = function(reconstruction, input, mean, log_var, embedding,
                                          lambda = 1) {
                   ridge <- torch_sum(embedding^2, dim = 2L) * lambda
                   kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - latent_dim
                   recon1 <- torch_sum(torch_square(input - reconstruction), dim = 2L) / torch_exp(self$loggamma)
                   recon2 <- self$env_vae$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$env_vae$input_dim
                   loss <- torch_mean(recon1 + recon2 + kl)
                   list(loss, torch_mean(recon1*torch_exp(self$loggamma)), torch_mean(kl), torch_mean(ridge))
                 },
  forward = function(x, c) {
    c_embedding <- self$embedder(c)

    # z <- env_vae$reparameterize(torch_zeros(x$size()[[1]], self$env_vae$latent_dim)$cuda(),
    #                             torch_zeros(x$size()[[1]], self$env_vae$latent_dim)$cuda())

    c(means, log_vars) %<-% self$env_vae$encoder(x, c_embedding)
    z <- self$env_vae$reparameterize(means, log_vars)
    # recon <- self$env_vae$decoder(z, c_embedding)
    # recon
    list(self$env_vae$decoder(z, c_embedding), x, c_embedding, means, log_vars)
  }
)

species_search <- species_search_mod(nlevels(unseen_dat$Label),
                                     env_vae,
                                     -10)
species_search <- species_search$cuda()

num_epochs <- 10000

lr <- 0.002
optimizer3 <- optim_adam(species_search$parameters, lr = lr)
scheduler3 <- lr_one_cycle(optimizer3, max_lr = lr,
                           epochs = num_epochs, steps_per_epoch = length(unseen_dl),
                           cycle_momentum = FALSE)

latent_dim <- env_vae$latent_dim

intermediate_zs <- list()
this_epoch <- 1
for (epoch in 1:num_epochs) {

    batchnum <- 0
    this_epoch <- 0
    coro::loop(for (b in unseen_dl) {

        batchnum <- batchnum + 1
        optimizer3$zero_grad()

        # recon <- species_search(b$env$cuda(), b$spec$cuda())
        # reconstruction_loss <- torch_mean((recon - b$env$cuda())^2) * 1000

        c(reconstruction, input, embedding, mean, log_var) %<-% species_search(b$env$cuda(), b$spec$cuda())
        c(loss, reconstruction_loss, kl_loss, ridge_loss) %<-% species_search$loss_function(reconstruction, input, mean, log_var, embedding)

        if(epoch > this_epoch) {

          intermediate_zs[[epoch]] <- as.matrix(species_search$embedder$weight$cpu())

        }

 #       if(batchnum %% 4 == 0) {

            cat("Epoch: ", epoch,
                "  batch: ", batchnum,
                "  loss: ", as.numeric(loss$cpu()),
                "  recon loss: ", as.numeric(reconstruction_loss$cpu()),
                "  KL loss: ", as.numeric(kl_loss$cpu()),
                "  loggamma: ", as.numeric(species_search$loggamma$cpu()),
                "  ridge loss: ", as.numeric(ridge_loss$cpu()),
                "\n")

 #       }
        loss$backward()
        optimizer3$step()
        scheduler3$step()
        this_epoch <- epoch
    })
}

torch_save(species_search, "data/species_search.to")
write_rds(intermediate_zs, "data/intermediate_zs.rds")

species_search <- torch_load("data/species_search.to")
species_search$cuda()

set.seed(12671)

unseens <- unique(test_dat %>%
                    filter(type == "whole") %>%
                    pull(Label))
uns <- sample(unseens, 1)

uns <- unseen_dat$Label[unseen_dat$Label == levels(unseen_dat$Label)[148]][1]
uns <- test_dat$Label[as.character(test_dat$Label) == as.character(uns)][1]

presence <- test_dat %>%
  filter(Label == uns)

pres_xy <- full_dat %>%
  filter(training == 0) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  filter(Label == uns)

plot(pres_xy[,2:3])

embed_final <- unseen_dat %>%
  filter(as.character(Label) == as.character(uns)) %>%
  pull(Label) %>%
  unique() %>%
  torch_tensor()
embed_final <- species_search$embedder(embed_final$cuda())

unseen_eval <- evaluate_species(embedding = list(embed_final), env_vae = env_vae,
                                xy_vae = xy_vae,
                                presence = presence, pres_xy = pres_xy,
                                background = background,
                                test = NULL)

unseen_eval$rf_mod$training.evaluation
unseen_eval$train_auc
unseen_eval$geo_plot
unseen_eval$env_plot


unseen_pres_list <- test_dat %>%
  filter(type == "whole") %>%
  group_by(Label) %>%
  group_nest(keep = TRUE)

for_unseen_dat <- unseen_pres_list %>%
  rename(unseen_env_data = data) %>%
  left_join(test_xy_list %>% rename(unseen_xy_data = data))

for_unseen_dat <- for_unseen_dat %>%
  rowwise() %>%
  mutate(n_unseen_xy = list(nrow(unseen_xy_data)),
         n_unseen_env = list(nrow(unseen_env_data))) %>%
  ungroup() %>%
  filter(!sapply(n_unseen_xy, is.null))

get_unseen_auc <- function(Label, env_vae,
                           xy_vae,
                           presence,
                           pres_xy,
                           background,
                           unseen_dat,
                           species_search) {

  embed_final <- unseen_dat %>%
    filter(as.character(Label) == as.character(Label[[1]])) %>%
    pull(Label) %>%
    unique() %>%
    torch_tensor()

  embed_final <- species_search$embedder(embed_final$cuda())

  unseen_eval <- evaluate_species(embedding = list(embed_final), env_vae = env_vae,
                                  xy_vae = xy_vae,
                                  presence = presence, pres_xy = pres_xy,
                                  background = background,
                                  do_plots = FALSE,
                                  do_rf = FALSE)

  unseen_eval
}

unseen_auc <- pmap(for_unseen_dat,
                ~ suppressMessages(get_unseen_auc(..1,
                                   env_vae = env_vae,
                                   xy_vae = xy_vae,
                                   presence = ..2,
                                   pres_xy = ..3,
                                   background = background,
                                   unseen_dat = unseen_dat,
                                   species_search = species_search)),
                .progress = TRUE)

write_rds(unseen_auc, "data/all_unseen_aucs.rds")


unseen_aucs <- map(unseen_auc, ~ .x$train_auc$.estimate[1]) %>%
  compact() %>%
  unlist()

auc_df <- for_unseen_dat %>%
  mutate(aucs = map(unseen_auc, ~ .x$train_auc$.estimate[1])) %>%
  mutate(aucs = unlist(aucs),
         n_unseen_xy = unlist(n_unseen_xy))

median(auc_df$aucs[auc_df$n_unseen_xy > 5])

plot(auc_df$n_unseen_xy, auc_df$aucs)

quantile(unseen_aucs, c(0.025, 0.5, 0.975), na.rm = TRUE)
mean(unseen_aucs, na.rm = TRUE)


########## umap niche embedding ###########
library(uwot)
conflict_prefer("points", "graphics")

train_spec_r <- training_dat$Label[which(training_dat$Label %in% unique(training_dat$Label))] %>%
  unique()
train_spec <- train_spec_r %>%
  torch_tensor()
niche_embedding <- as.matrix(env_vae$embedder(train_spec$cuda())$cpu())

niche_umap <- umap(niche_embedding, 5, ret_model = TRUE)
plot(niche_umap$embedding)

niche_embedding_unseen <- as.matrix(species_search$embedder$weight$cpu())
niche_umap_unseen <- umap_transform(niche_embedding_unseen, niche_umap)
plot(niche_umap$embedding, pch = 19)
points(niche_umap_unseen, pch = 19, col = "red")
text(niche_umap_unseen, labels = 1:nrow(niche_embedding_unseen), col = "blue")

uns2 <- unseen_dat$Label[unseen_dat$Label == levels(unseen_dat$Label)[148]][1]
uns2 <- test_dat$Label[as.character(test_dat$Label) == as.character(uns2)][1]

presence2 <- test_dat %>%
  filter(Label == uns)

pres_xy2 <- full_dat %>%
  filter(training == 0) %>%
  select(Label, x = Longitude, y = Latitude) %>%
  filter(Label == uns)

intermediate_zs <- read_rds("data/intermediate_zs.rds")

zs2 <- map(intermediate_zs, ~.x[148,]) %>%
  do.call(rbind, .)

niche_umap_search <- umap_transform(zs2, niche_umap)

plot(niche_umap$embedding, pch = 19)
points(niche_umap_unseen, pch = 19, col = "red")
points(niche_umap_search, type = "l", col = "blue")

train_umap <- as.data.frame(niche_umap$embedding) %>%
  mutate(data = "training species")
test_umap <- as.data.frame(niche_umap_unseen) %>%
  mutate(data = "testing species")
umap_df <- rbind(train_umap, test_umap)
search_umap <- as.data.frame(niche_umap_search)

pp <- ggplot(umap_df, aes(V1, V2)) +
  geom_point(aes(colour = data), size = 2.5) +
  geom_path(data = search_umap[c(FALSE, FALSE, FALSE,
                                 FALSE, FALSE, FALSE,
                                 TRUE), ],
            linetype = 1, colour = "grey30") +
  geom_point(data = test_umap, aes(colour = data), size = 3, alpha = 0.7) +
  geom_point(data = search_umap[c(FALSE, FALSE, FALSE,
                                 FALSE, FALSE, FALSE,
                                 TRUE), ],
             size = 0.2) +
  scale_color_discrete(name = "") +
  xlab("Niche Embedding UMAP Axis 1") +
  ylab("Niche Embedding UMAP Axis 2") +
  theme_minimal() +
  theme(legend.position = c(0.8, 0.8),
        legend.text = element_text(size = 24),
        axis.title = element_text(size = 26),
        axis.text = element_text(size = 18))


samps <- floor(c(0, 0.05, 0.12, 0.45, 0.6, 1) * 10000)
samps[1] <- 1
plot(niche_umap_search[samps, ])

pp + geom_point(data = as.data.frame(niche_umap_search[samps, ]), col = "white", size = 3) +
  geom_text(label = 1:6, data = as.data.frame(niche_umap_search[samps, ]), col = "white",
            nudge_x = 0.3, size = 6)

p1s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[1], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

p2s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[2], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

p3s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[3], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

p4s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[4], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

p5s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[5], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

p6s <- unseen_eval <- evaluate_species(embedding = list(torch_tensor(zs2[samps[6], , drop = FALSE])$cuda()),
                                       env_vae = env_vae,
                                       xy_vae = xy_vae,
                                       presence = presence2, pres_xy = pres_xy2,
                                       background = background,
                                       do_rf = FALSE)

library(patchwork)
(p1s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  (p2s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  (p3s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  (p4s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  (p5s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  (p6s$geo_plot + scale_fill_viridis_c(option = "inferno", trans = "sqrt",
                                    limits = c(0, 1)) + theme_void()) +
  plot_layout(ncol = 3, nrow = 2, guides = "collect") +
  plot_annotation(tag_levels = '1',
                  tag_suffix = ".")



