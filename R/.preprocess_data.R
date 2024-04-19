library(jsonlite)
library(taxize)
library(tidyverse)

############ GeoLifeCLEF ################
geo_train <- read_json("data/GeoLifeCLEF2020/annotations_train.json")
geo_val <- read_json("data/GeoLifeCLEF2020/annotations_val.json")

geo_dat <- geo_train$images %>%
  list_transpose() %>%
  as_tibble()

cat_dat <- geo_train$categories %>%
  list_transpose() %>%
  as_tibble()

anno_dat <- geo_train$annotations %>%
  list_transpose() %>%
  as_tibble()

geo_dat <- geo_dat %>%
  left_join(anno_dat %>%
              select(id = image_id, category_id)) %>%
  left_join(cat_dat %>%
              select(category_id = id, gbif_id, gbif_name))

geo_dat2 <- geo_val$images %>%
  list_transpose() %>%
  as_tibble()

cat_dat2 <- geo_val$categories %>%
  list_transpose() %>%
  as_tibble()

anno_dat2 <- geo_val$annotations %>%
  list_transpose() %>%
  as_tibble()

geo_dat2 <- geo_dat2 %>%
  left_join(anno_dat2 %>%
              select(id = image_id, category_id)) %>%
  left_join(cat_dat %>%
              select(category_id = id, gbif_id, gbif_name))

geo_dat <- geo_dat %>%
  bind_rows(geo_dat2)

write_csv(geo_dat, "data/GeoLifeCLEF2020/geo_dat_raw.csv")

gbif_ids <- unique(geo_dat$gbif_id)

gbif_dat <- map(gbif_ids,
                possibly(insistently(~ gbif_name_usage(.x),
                                     rate_backoff(max_times = 4)),
                         otherwise = NULL,
                         quiet = FALSE),
                .progress = TRUE)

write_rds(gbif_dat, "data/GeoLifeCLEF2020/gbif_taxa_info.rds")

gbif_df <- gbif_dat %>%
  list_transpose() %>%
  as_tibble()

geo_dat <- geo_dat %>%
  left_join(gbif_df %>%
              select(gbif_id = key,
                     kingdom,
                     family,
                     genus)) %>%
  filter(kingdom == "Plantae")

write_csv(geo_dat, "data/GeoLifeCLEF2020/geo_dat_plants.csv")

######### Oz Plants ##############################

oz_files <- list.files("data/Oz", full.names = TRUE)

oz_dat <- map(oz_files,
              ~ suppressWarnings(read_csv(.x)) %>%
                drop_na(coordinateUncertaintyInMeters, year,
                        decimalLatitude, decimalLongitude) %>%
                filter(coordinateUncertaintyInMeters < 30,
                       year > 2017, year < 2023) %>%
                select(decimalLongitude, decimalLatitude,
                       species, genus, family,
                       scientificName),
              .progress = TRUE)

oz_dat <- oz_dat %>%
  list_rbind()


tt <- suppressWarnings(read_csv(oz_files[[1]])) %>%
                drop_na(coordinateUncertaintyInMeters, year,
                        decimalLatitude, decimalLongitude) %>%
                filter(coordinateUncertaintyInMeters < 30,
                       year > 2017, year < 2023) %>%
                select(decimalLongitude, decimalLatitude,
                       species, genus, family,
                       scientificName)
