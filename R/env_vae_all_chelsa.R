library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(terra)
library(SharedObject)
library(future)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(536567678)