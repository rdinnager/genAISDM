library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(567678)

squamate_train <- read_csv("output/squamate_training2.csv")

bino <- squamate_train$Binomial

species <- as.integer(as.numeric(as.factor(squamate_train$Binomial)))

squamate_train <- squamate_train |>
  select(-Binomial, -X, -Y, -n) |>
  as.matrix()
#squamate_test <- read_csv("output/squamate_testing.csv")
scaling <- read_rds("output/squamate_env_scaling2.rds")

squamate_train <- scale(squamate_train,
                        center = unlist(scaling$means), scale = unlist(scaling$sd))

options(torch.serialization_version = 2)
env_vae <- torch_load("data/env_vae_trained_fixed2_alpha_0.5_32d.to")
env_vae <- env_vae$cuda()

# env_vae2 <- torch_load("data/env_vae_1_trained_fixed_16d_stage2.to")
# env_vae2 <- env_vae2$cuda()

na_mask <- apply(squamate_train, 2, function(x) as.numeric(!is.na(x)))

squamate_train[na_mask == 0] <- 0

env_dataset <- dataset(name = "env_ds",
                       initialize = function(env, mask, spec) {
                         self$env <- torch_tensor(env)
                         self$mask <- torch_tensor(mask)
                         self$spec <- torch_tensor(spec)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, ], mask = self$mask[i,], spec = self$spec[i])
                       },
                       .length = function() {
                         self$env$size()[[1]]
                       })

train_ds <- env_dataset(squamate_train, na_mask, species)
train_dl <- dataloader(train_ds, 900000, shuffle = FALSE)

code_list <- list()
var_list <- list()
spec_list <- list()
code2_list <- list()
i <- 0
coro::loop(for (b in train_dl) {
  with_no_grad( {
    i <- i + 1
    s <- env_vae$species_embedder_mean(b$spec$cuda())
    codes <- env_vae$encoder(b$env$cuda(), s)
    #codes2 <- env_vae2$encoder(codes$means, s)
    code_list[[i]] <- as.matrix(codes$means$cpu())
    var_list[[i]] <- as.matrix(codes$logvars$cpu())
    #code2_list[[i]] <- as.matrix(codes2$means$cpu())
    spec_list[[i]] <- as.matrix(s$cpu())
  })
  print(i)
})

codes <- do.call(rbind, code_list)
#codes2 <- do.call(rbind, code2_list)
specs <- do.call(rbind, spec_list)
vars <- do.call(rbind, var_list)

mani <- which(apply(exp(vars), 2, mean) < 0.5)
mani

write_rds(mani, "output/squamate_env_mani2.rds")

colnames(specs) <- paste0("L", 1:ncol(specs))

codes_df <- as.data.frame(codes[ , mani]) |>
  bind_cols(as.data.frame(specs)) |>
  mutate(species = bino)


write_rds(codes_df, "output/squamate_env_latent_codes_for_stage2_2.rds")
#write_rds(test_codes, "output/squamate_env_latent_codes_for_stage2_testing.rds")

# codes2_df <- as.data.frame(codes2) |>
#   mutate(species = specs)
# 
# vars1 <- apply(codes, 2, var)
# vars2 <- apply(codes2, 2, var)
# 
# var2_choose <- which(vars2 > 0.1)
# 
# rand_spec <- sample(specs, 1)
# #rand_spec <- "Varanus caudolineatus"
# 
# codes2_spec <- codes2_df |>
#   filter(species == rand_spec)
# 
# qqnorm(codes2_spec[ , var2_choose[1]])
# qqline(codes2_spec[ , var2_choose[1]])
# 
# qqnorm(codes2_spec[ , var2_choose[2]])
# qqline(codes2_spec[ , var2_choose[2]])
# 
# qqnorm(codes2_spec[ , var2_choose[3]])
# qqline(codes2_spec[ , var2_choose[3]])
# 
# hist(codes[ , which(vars1 > 0.1)[1]])
# hist(codes2[ , which(vars2 > 0.1)[1]])
# 
# hist(codes[ , which(vars1 > 0.1)[2]])
# hist(codes2[ , which(vars2 > 0.1)[2]])
# 
# hist(codes[ , which(vars1 > 0.1)[3]])
# hist(codes2[ , which(vars2 > 0.1)[3]])
# 
# qqnorm(codes[sample.int(nrow(codes), 100000) , which(vars1 > 0.1)[1]])
# qqline(codes[sample.int(nrow(codes), 100000) , which(vars1 > 0.1)[1]])
# qqnorm(codes2[sample.int(nrow(codes2), 100000) , which(vars2 > 0.1)[1]])
# qqline(codes2[sample.int(nrow(codes2), 100000) , which(vars2 > 0.1)[1]])
# 
# write_rds(list(codes = codes, specs = specs), "output/squamate_env_codes.rds")