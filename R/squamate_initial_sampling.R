library(tidyverse)
library(sf)
library(terra)
library(furrr)
library(rnaturalearth)

land <- ne_download(scale = 10, type = 'land', category = 'physical')

squamates2 <- read_rds("data/final_squamate_sf.rds")

#polyg <- squamates2$geometry[1]
sample_points <- function(polyg_num, polyg, n = 10000,
                          folder = "output/squamate_samples2") {
  
  fp <- file.path(folder, paste0(str_pad(polyg_num, 5, pad = "0"), ".rds"))
  
  if(file.exists(fp)) {
    return(fp)
  }
  
  ## placemarker file
  write_rds(st_sfc(), fp)
  
  if(st_is_valid(polyg)) {
    
    samp <- st_sample(polyg, n)  
    
    coords <- samp |> st_coordinates()
    
    # xrange <- range(coords[ , 1])
    # yrange <- range(coords[ , 2])
    # xexpand <- (xrange[2] - xrange[1]) / 2
    # yexpand <- (yrange[2] - yrange[1]) / 2
    # 
    # xrange <- xrange + c(-xexpand, xexpand)
    # yrange <- yrange + c(-yexpand, yexpand)
    # 
    # dens <- MASS::kde2d(coords[ , 1], coords[ , 2], n = 100,
    #                     lims = c(xrange, yrange))
    # 
    # dens_img <- as.cimg(dens$z)
    # dens_img <- isoblur(dens_img, sigma = 5)
    # 
    # dens$z <- as.matrix(dens_img)
    # 
    # bands <- isobands(dens$x, dens$y, t(dens$z / max(dens$z)), seq(0, 0.9, by = 0.1), seq(0.1, 1.0, by = 0.1))
    # bsf <- st_as_sfc(iso_to_sfg(bands))
    # 
    # bsf <- bsf[!st_is_empty(bsf)]
    # bsf <- bsf[-1]
    # bsf <- rev(bsf)
    # bsf_areas <- st_area(bsf)
    
    # probs <- seq(0, 1, length.out = length(bsf))
    # denses <- dnorm(probs, sd = norm_sd) 
    # denses <- denses * bsf_areas
    # denses <- (denses / sum(denses)) * n
    # 
    # samps <- map2(bsf, ceiling(denses), ~ st_sample(.x, .y))
    # samps <- do.call(c, samps)
    
    st_crs(samp) <- st_crs(polyg)
    
    samp <- st_intersection(samp, land$geometry)
    
    write_rds(samp, fp)
    
    
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

