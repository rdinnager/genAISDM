library(tidyverse)
library(sf)
library(ape)
library(rnaturalearth)
library(terra)
library(isoband)
library(imager)
library(furrr)

ocean <- ne_download(scale = 10, type = 'ocean', category = 'physical')
land <- ne_download(scale = 10, type = 'land', category = 'physical')

ocean <- st_make_valid(ocean)

squamates <- st_read("data/maps/squamates/squamates")
#tree <- read.tree("data/phylogenies/squamates/squam_shl_new_Posterior_9755.5000.trees")[[101]]
#write.tree(tree, "data/phylogenies/squamates/squamates.tre")

tree <- read.tree("data/phylogenies/squamates/squamates.tre")

n_intree <- sum(gsub(" ", "_", squamates$Binomial) %in% tree$tip.label)

# invalid_geoms <- !st_is_valid(squamates$geometry)
# sum(invalid_geoms)
# squamates$geometry <- 

#sf_use_s2(FALSE)
#squamates <- st_make_valid(squamates)
#invalid_geoms <- !st_is_valid(squamates$geometry)
#squa_union <- st_union(squamates$geometry)
#plot(squa_union)

#sf_use_s2(TRUE)

squamates2 <- squamates |>
  filter(gsub(" ", "_", Binomial) %in% tree$tip.label)

write_rds(squamates2, "data/final_squamate_sf.rds")

#polyg <- squamates2$geometry[1]
sample_points <- function(polyg_num, polyg, split_into = 10,
                          norm_sd = 0.4, n = 1000,
                          folder = "output/squamate_samples") {
  
  fp <- file.path(folder, paste0(str_pad(polyg_num, 5, pad = "0"), ".rds"))
  
  if(file.exists(fp)) {
    return(fp)
  }
  
  ## placemarker file
  write_rds(st_sfc(), fp)
  
  if(st_is_valid(polyg)) {
    
    samp <- st_sample(polyg, 10000)  
    
    coords <- samp |> st_coordinates()
    
    xrange <- range(coords[ , 1])
    yrange <- range(coords[ , 2])
    xexpand <- (xrange[2] - xrange[1]) / 2
    yexpand <- (yrange[2] - yrange[1]) / 2
    
    xrange <- xrange + c(-xexpand, xexpand)
    yrange <- yrange + c(-yexpand, yexpand)
    
    dens <- MASS::kde2d(coords[ , 1], coords[ , 2], n = 100,
                        lims = c(xrange, yrange))
    
    dens_img <- as.cimg(dens$z)
    dens_img <- isoblur(dens_img, sigma = 5)
    
    dens$z <- as.matrix(dens_img)
    
    bands <- isobands(dens$x, dens$y, t(dens$z / max(dens$z)), seq(0, 0.9, by = 0.1), seq(0.1, 1.0, by = 0.1))
    bsf <- st_as_sfc(iso_to_sfg(bands))
    
    bsf <- bsf[!st_is_empty(bsf)]
    bsf <- bsf[-1]
    bsf <- rev(bsf)
    bsf_areas <- st_area(bsf)
    
    probs <- seq(0, 1, length.out = length(bsf))
    denses <- dnorm(probs, sd = norm_sd) 
    denses <- denses * bsf_areas
    denses <- (denses / sum(denses)) * n
    
    samps <- map2(bsf, ceiling(denses), ~ st_sample(.x, .y))
    samps <- do.call(c, samps)
    
    st_crs(samps) <- st_crs(polyg)
    
    samps <- st_intersection(samps, land$geometry)
    
    write_rds(samps, fp)
    
    
  } else {
    fp <- ""
  }
  
  fp
  
}

future::plan(future::multisession())

samples <- future_map(seq_along(squamates2$geometry),
                      possibly(~ sample_points(.x, squamates2$geometry[.x]),
                               ""),
                      .progress = TRUE)

