library(tidyverse)
library(sf)
library(terra)
library(furrr)
library(rnaturalearth)
library(fasterize)

land <- ne_download(scale = 10, type = 'land', category = 'physical')

chelsa_files <- list.files("data/env/CHELSA-BIOMCLIM+/1981-2010/bio", 
                           full.names = TRUE)

chelsa <- rast(chelsa_files)

chelsa_samp <- spatSample(chelsa, size = c(5220, 10800),
                          method = "regular",
                          as.raster = TRUE,
                          xy = TRUE)

land_rast <- fasterize(land, raster::raster(chelsa_samp))
land_rast <- rast(land_rast)

rm(chelsa)
gc()

chelsa_samp <- mask(chelsa_samp, land_rast)

gc()

writeRaster(chelsa_samp, "output/rasts/chelsa_masked_downsampled.tif")

gc()

chelsa_samp <- rast("output/rasts/chelsa_masked_downsampled.tif")

chelsa_df <- as.data.frame(chelsa_samp, xy = TRUE)

chelsa_df <- chelsa_df |>
  select(-contains("kg"), -contains("lgd"),
         -contains("fgd"))

write_rds(chelsa_df, "output/geo_env_chelsa.rds")
