source("R/load_and run_models.R")
source("R/geo_utils_and_data.R")
source("R/plotting.R")

library(tidymodels)
library(probably)
library(tidysdm)

library(rayshader)
library(patchwork)

env_vae <- load_nichencoder_vae()
flow_2 <- load_nichencoder_flow_rectified()
geode <- load_geode_flow()