library(sf)
library(h3)

find_equal_area_projection <- function(sf_object) {
  # Ensure the dataset is in WGS84 geographic coordinates
  sf_object <- st_transform(sf_object, 4326)
  
  # Calculate the bounding box and centroid
  bbox <- st_bbox(sf_object)
  centroid <- st_coordinates(st_centroid(st_union(sf_object)))
  lon0 <- centroid[1]
  lat0 <- centroid[2]
  
  # Calculate extent in degrees
  delta_lon <- bbox$xmax - bbox$xmin
  delta_lat <- bbox$ymax - bbox$ymin
  
  # Determine if the dataset has a global extent
  is_global_extent <- delta_lon >= 180 || delta_lat >= 90
  
  if (is_global_extent) {
    # Use Equal Earth projection for global datasets
    proj_str <- "+proj=eqearth +units=m +ellps=WGS84"
  } else if (delta_lat > delta_lon) {
    # Predominantly north-south extent
    # Use Lambert Azimuthal Equal-Area projection
    proj_str <- sprintf(
      "+proj=laea +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      lat0, lon0
    )
  } else {
    # Predominantly east-west extent
    # Use Albers Equal-Area Conic projection
    # Set standard parallels based on dataset's latitude
    std_parallel_1 <- lat0 - delta_lat / 6
    std_parallel_2 <- lat0 + delta_lat / 6
    proj_str <- sprintf(
      "+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      std_parallel_1, std_parallel_2, lat0, lon0
    )
  }
  
  # Reproject the data
  #sf_projected <- st_transform(sf_object, crs = proj_str)
  
  return(proj_str)
}

localize <- function(coord_predict, spec, ecoregions, world, use_preds = FALSE, res = 5) {
  
  pred_sf <- coord_predict$predictions |>
    as.data.frame() |>
    st_as_sf(coords = c("V1", "V2"), crs = 4326)
  truth_sf <- coord_predict$truth |>
    st_as_sf(coords = c("X", "Y"), crs = 4326)
  if(use_preds) {
    samp_coords <- rbind(pred_sf,
                         truth_sf) |>
      mutate(point = 1)  
  } else {
    samp_coords <- truth_sf |>
      mutate(point = 1)
  }
  
  hex_res <- map(1:10, ~ unique(geo_to_h3(truth_sf, .x)), .progress = TRUE)
  n_hexes <- map_int(hex_res, length)
  res_choice <- which.min(abs(300 - n_hexes))
  hexes <- hex_res[[res_choice]]
  hex_sf <- h3_to_geo_boundary_sf(hexes)
  
  ecoreg <- ecoregions |>
    st_join(hex_sf) |>
    filter(!is.na(h3_index)) |>
    distinct(ECO_NAME, .keep_all = TRUE)
  # ecoreg <- ecoreg |>
  #   st_cast("POLYGON") |>
  #   select(-h3_index) |>
  #   mutate(poly_id = 1:n())
  # ecoreg <- ecoreg |>
  #   st_join(hex_sf) |>
  #   filter(!is.na(h3_index)) |>
  #   distinct(poly_id, .keep_all = TRUE)
  
  hex_points <- pred_sf |>
    mutate(point = 1) |>
    st_join(hex_sf) |>
    group_by(h3_index)
  
  hex_counts_pred <- hex_points |>
    summarise(count = sum(point)) |>
    ungroup() |>
    mutate(prop = count / max(count)) |>
    filter(prop > 0.01) 
  
  # pred_sf_red <- hex_points |>
  #   mutate(count = sum(point)) |>
  #   ungroup() |>
  #   mutate(prop = count / max(count)) |>
  #   filter(prop > 0.01) 
  # test <- st_kcde(pred_sf,  verbose = TRUE)
  # kde_contour <- st_get_contour(test)
  
  hex_polys <- hex_sf |>
    left_join(hex_counts_pred |> as_tibble() |> select(-geometry))
  
  hex_polys <- hex_polys |>
    st_intersection(ecoreg)
  
  proj <- find_equal_area_projection(ecoreg)
  
  ecoreg <- ecoreg |>
    st_transform(proj)
  countries <- world |>
    st_transform(proj)
  hex_polys <- hex_polys |>
    st_transform(proj)
  
  extent <- st_bbox(ecoreg)
  hex_polys <- hex_polys |>
    mutate(prop = count / sum(count, na.rm = TRUE))
  
  
  list(pred_sf = pred_sf |> st_transform(proj), 
       truth_sf = truth_sf |> st_transform(proj), 
       hex_polys = hex_polys,
       ecoregions = ecoreg, 
       countries = countries,
       extent = extent)
  
}
