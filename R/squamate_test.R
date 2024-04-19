library(tidyverse)
library(sf)
library(ape)

squamates <- st_read("data/maps/squamates/squamates")
#tree <- read.tree("data/phylogenies/squamates/squam_shl_new_Posterior_9755.5000.trees")[[101]]
#write.tree(tree, "data/phylogenies/squamates/squamates.tre")

tree <- read.tree("data/phylogenies/squamates/squamates.tre")

n_intree <- sum(gsub(" ", "_", squamates$Binomial) %in% tree$tip.label)