library(tidyverse)
library(sf)
library(rnaturalearth)
library(torch)
library(zeallot)

chelsa_df <- read_rds("output/geo_env_chelsa.rds")
codes_df <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds")
maps <- read_rds("data/final_squamate_sf.rds")
world <- ne_coastline()
#ecoregions <- read_sf("data/maps/ecoregions")

# ecoregions <- ecoregions |>
#   st_make_valid()
# write_rds(ecoregions, "data/maps/ecoregions_valid.rds")

ecoregions <- read_rds("data/maps/ecoregions_valid.rds")

spec_codes <- codes_df |>
  select(species, starts_with("L")) |>
  distinct(species, .keep_all = TRUE)

scaling <- read_rds("output/squamate_env_scaling.rds")

chelsa_sf <- chelsa_df |>
  select(x, y) |>
  st_as_sf(coords = c("x", "y"), crs = 4326) |>
  st_join(ecoregions)  
ecoreg_df <- chelsa_df |>
  select(x, y) |>
  bind_cols(chelsa_sf |>
              as_tibble() |>
              select(-geometry))
write_rds(ecoreg_df, "data/maps/ecoreg_chelsa_df.rds")

env <- chelsa_df |>
  select(-x, -y) |>
  as.matrix()

m <- scaling$means[colnames(env)]
s <- scaling$sd[colnames(env)]

env <- t((t(env) - m) / s)

na_mask <- apply(env, 2, function(x) as.numeric(!is.na(x)))

env[na_mask == 0] <- 0

env_dataset <- dataset(name = "env_ds",
                       initialize = function(env, mask) {
                         self$env <- torch_tensor(env)
                         self$mask <- torch_tensor(mask)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, ], mask = self$mask[i,])
                       },
                       .length = function() {
                         self$env$size()[[1]]
                       })

train_ds <- env_dataset(env, na_mask)
train_dl <- dataloader(train_ds, 900000, shuffle = TRUE)

test <- train_dl$.iter()$.next()

options(torch.serialization_version = 2)
env_vae <- torch_load("data/env_vae_trained_fixed2_alpha_0.5_32d.to")
env_vae <- env_vae$cuda()

species <- spec_codes$species[4]

b <- test
spec <- species
calc_elbo <- function(env_vae, b, spec, spec_codes, loggamma = NULL) {
  
  if(is.null(loggamma)) {
    loggamma <- env_vae$loggamma
  } else {
    loggamma <- torch_tensor(loggamma, device = "cuda")
  }
  
  with_no_grad({
    spec <- spec_codes |>
      filter(species == spec) |>
      select(-species) |>
      as.matrix() 
    spec <- matrix(spec, nrow = b$env$size()[1], ncol = length(spec), byrow = TRUE) |>
      torch_tensor()
    c(means, log_vars) %<-% env_vae$encoder(y = b$env$cuda(), s = spec$cuda())
    z <- env_vae$reparameterize(means, log_vars)
    
    kl <- torch_sum(torch_exp(log_vars) + torch_square(means) - log_vars, dim = 2L) - env_vae$latent_dim
    reconstruction <- env_vae$decoder(z, spec$cuda())
    recon1 <- torch_sum(torch_square(b$env$cuda() - reconstruction) * b$mask$cuda(), dim = 2L) / torch_exp(loggamma)
    recon2 <- env_vae$input_dim * loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * env_vae$input_dim
    
    elbo <- kl + recon1 + recon2
    
  })
  return(-elbo$cpu())
}

rep_1 <- list()
i <- 0
coro::loop(for(b in train_dl) {
  i <- i + 1
  rep_1[[i]] <- calc_elbo(env_vae, b, spec, spec_codes, loggamma = 1)
  print(i)
})
rep_1 <- torch_cat(rep_1)

elbo_df <- chelsa_df |>
  select(x, y) |>
  mutate(elbo = as.numeric(rep_1),
         prob = exp(elbo))

cairo_pdf(width = 20, height = 10)
ggplot(elbo_df, aes(x, y)) +
  geom_raster(aes(fill = prob)) +
  scale_fill_continuous(trans = 'log1p') +
  theme_void()
dev.off()

test_out <- calc_elbo(env_vae, b, spec, spec_codes, loggamma = 1)

test_out_mat <- as.matrix(test_out$cpu())

species <- spec_codes$species[4]

spec_geo <- maps |> 
  filter(Binomial == species) |>
  st_make_valid()
ecoreg_overlap <- st_join(ecoregions, spec_geo) |>
  filter(!is.na(Binomial), ECO_NAME != "Rock and Ice")

elbo_sf <- st_as_sf(elbo_df, coords = c("x", "y"), crs = 4326)

elbo_cut <- elbo_sf |>
  st_join(ecoreg_overlap)
elbo_cut <- elbo_cut |>
  filter(!is.na(ECO_NAME))

elbo_coord_df <- elbo_cut |>
  st_coordinates() |>
  as.data.frame() |>
  mutate(prob = elbo_cut$prob)

elbo_high <- elbo_df |>
  filter(prob > 1e-59)

plot(elbo_high |> select(x, y))

ggplot(ecoreg_overlap) + 
  geom_sf() +
  geom_sf(data = spec_geo, fill = "red") #+
  #geom_raster(aes(X, Y, fill = prob), data = elbo_coord_df) +
  #scale_fill_continuous(trans = 'log1p')

plot(ecoreg_overlap |> select(ECO_NAME))

plot(ecoreg_overlap |> select(geometry))
plot(spec_geo |> select(geometry), add = TRUE,
     col = "red")

plot(spec_geo |> select(geometry),
     col = "red", lwd = 1)