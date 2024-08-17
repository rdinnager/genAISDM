library(tidyverse)
library(gifski)
library(patchwork)

plots <- read_rds("output/niche_interpolate_1.rds")

test1 <- plots$niche_plots[[1]]
test2 <- plots$latent_plots[[1]]

test2 + test1 + plot_layout(ncol = 2, widths = c(0.4, 0.6))

plot_plots <- function() {
  walk2(plots$niche_plots, plots$latent_plots, 
        ~ plot(.y + .x + plot_layout(ncol = 2, widths = c(0.35, 0.65))))
}

gifski::save_gif(plot_plots(), "figures/interp_animation_1_bigger.gif",
                 delay = 0.06, progress = TRUE,
                 height = 1000, width = 1600)

utils::browseURL("figures/interp_animation_1.gif")