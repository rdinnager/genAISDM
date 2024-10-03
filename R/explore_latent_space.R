library(tidyverse)
library(uwot)
library(scico)
library(patchwork)

set.seed(852348)

lats <- read_csv("output/squamate_latitudes2.csv")
areas <- read_csv("output/squamate_areas.csv")

# write_csv(lats |> left_join(areas),
#           "output/species_meta.csv")

dat <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  distinct(species, .keep_all = TRUE)
spec_lats <- dat |>
  select(species, starts_with("L"))

dat_all <- spec_lats |>
  left_join(lats, by = c("species" = "Binomial")) |>
  left_join(areas, by = c("species" = "Binomial")) |>
  mutate(abs_lat = abs(mid_lat))

latent_df <- dat_all |>
  select(starts_with("L"))
niche_umap <- umap(latent_df, n_neighbors = 25, ret_model = TRUE) 

dat_all <- dat_all |>
  bind_cols(niche_umap$embedding |>
              as.data.frame() |>
              rename(UMAP_1 = V1, UMAP_2 = V2))

inset_1_box <- c(xmin = -2.5, xmax = 2.5, ymin = -7.5, ymax = -2.5)
inset_1 <- ggplot(dat_all |> filter(UMAP_1 > inset_1_box[1] & 
                                      UMAP_1 < inset_1_box[2] & 
                                      UMAP_2 < inset_1_box[4] & 
                                      UMAP_2 > inset_1_box[3]), 
                  aes(UMAP_1, UMAP_2)) +
  geom_point(aes(colour = abs_lat, size = Area), alpha = 0.5) +
  scale_colour_scico(palette = "hawaii", name = "Absolute Median Latitude") +
  scale_size_area(trans = "sqrt", labels = label_number(scale = 1/1000),
                  name = "Geographic Area (km^2)") +
  theme_minimal() +
  theme(legend.position = 'none',
        plot.background = element_rect(colour = "pink", fill = "white", linewidth = 2),
        axis.title = element_blank())
inset_1

inset_2_box <- c(xmin = -7.5, xmax = -4.5, ymin = 0, ymax = 6)
inset_2 <- ggplot(dat_all |> filter(UMAP_1 > inset_2_box[1] & 
                                      UMAP_1 < inset_2_box[2] & 
                                      UMAP_2 < inset_2_box[4] & 
                                      UMAP_2 > inset_2_box[3]), 
                  aes(UMAP_1, UMAP_2)) +
  geom_point(aes(colour = abs_lat, size = Area), alpha = 0.5) +
  scale_colour_scico(palette = "hawaii", name = "Absolute Median Latitude") +
  scale_size_area(trans = "sqrt", labels = label_number(scale = 1/1000),
                  name = "Geographic Area (km^2)") +
  theme_minimal() +
  theme(legend.position = 'none',
        plot.background = element_rect(colour = "turquoise", fill = "white", linewidth = 2),
        axis.title = element_blank())
inset_2

main <- ggplot(dat_all, aes(UMAP_1, UMAP_2)) +
  geom_point(aes(colour = abs_lat, size = Area), alpha = 0.5) +
  annotate("rect", xmin = inset_1_box[1], xmax = inset_1_box[2],
           ymin = inset_1_box[3], ymax = inset_1_box[4],
           colour = "pink", fill = NA) +
  annotate("rect", xmin = inset_2_box[1], xmax = inset_2_box[2],
           ymin = inset_2_box[3], ymax = inset_2_box[4],
           colour = "turquoise", fill = NA) +
  scale_colour_scico(palette = "hawaii", name = "Absolute Median\nLatitude") +
  scale_size_area(trans = "sqrt", labels = label_number(scale = 1/1000),
                  name = "Geographic\nArea (km^2)") +
  theme_minimal() +
  theme(legend.position.inside = c(0.935, 0.35),
        legend.position = "inside",
        legend.box.background = element_rect(colour = "black", fill = "white"))
main

ragg::agg_png("figures/latent_niche_space.png", width = 1800, height = 1000,
              scaling = 2.5)
p <- main + inset_element(inset_1, 0, 0, 0.4, 0.4) + 
  inset_element(inset_2, 0, 0.6, 0.3, 1)
plot(p)
dev.off()


########### zeroshot embedding ##########
example_dat <- read_rds("output/model_results_zeroshot/Geophis bicolor_data.pt")
traj <- example_dat$zeroshot$trajectory
start <- traj$stage_2[[1]][1, ]
compar <- map(traj$stage_1, ~ t(.x) - start)
compar_sums <- map(compar, ~ apply(.x, 2, sum)) 
w <- which(map_lgl(compar_sums, ~ any(abs(.x) < 0.000001)))
wher <- which.min(compar_sums[[w]])

start_traj <- traj$stage_1[[w]][1:wher, ]
rest_traj <- map(traj[-1], ~ do.call(rbind, .x))
rest_traj <- do.call(rbind, rest_traj)

rest_traj <- rest_traj[c(TRUE, FALSE, FALSE, FALSE), ]
plot(rest_traj[,1:2], type = "l")

all_umap <- umap(latent_df |> as.matrix() |> rbind(rest_traj), n_neighbors = 25, ret_model = TRUE)
latent_df2 <- latent_df |>
  bind_cols(all_umap$embedding[1:8304, ] |> as.data.frame())

traj_df <- all_umap$embedding[8304:nrow(all_umap$embedding), ] |> as.data.frame()

traj_umap <- umap_transform(rest_traj, niche_umap)
plot(traj_umap, type = "l")


