library(tidyverse)
library(sf)
library(terra)
library(furrr)

sample_files <- list.files("output/squamate_samples", full.names = TRUE)

samples <- map(sample_files, read_rds, .progress = TRUE)
nums <- gsub(".rds", "", basename(sample_files)) |>
  parse_number()

chelsa_files <- list.files("data/env/CHELSA-BIOMCLIM+/1981-2010/bio", 
                           full.names = TRUE)
#chelsa <- rast()

pts <- samples[[1]]
extract_env <- function(pts, chelsa_files) {
  if(inherits(pts, "sfc")) {
    pt_sf <- st_as_sf(pts)
    if(nrow(pt_sf) == 0) {
      return(tibble())
    }
    chelsa <- rast(chelsa_files)
    env_df <- terra::extract(chelsa, pt_sf, ID = FALSE) 
    coord_df <- st_coordinates(pt_sf)
    env_df <- bind_cols(coord_df, env_df)
  } else {
    tibble()  
  }
}

future::plan(future::multisession(workers = 10))
env_dat <- future_map(samples, ~ extract_env(.x, chelsa_files = chelsa_files), .progress = TRUE)

write_rds(env_dat, "output/squamate_all_env_dat_w_coords.rds")

#env_dat <- read_rds("output/squamate_all_env_dat.rds")

squamates2 <- read_rds("data/final_squamate_sf.rds")
env_list <- replicate(nrow(squamates2), tibble(), simplify = FALSE)
env_list[nums] <- env_dat

nonempty <- map_lgl(env_list, ~ nrow(.x) > 0)

squamates2 <- squamates2 |>
  mutate(env_samps = env_list)

squamates2 <- squamates2 |>
  filter(nonempty)

write_rds(squamates2, "output/squamate_final_data_w_env_samps.rds")



