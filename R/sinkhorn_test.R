# Install torch if not already installed
# install.packages("torch")

library(torch)

cost_matrix <- function(x, y) {
  # x: Tensor of shape (n, d)
  # y: Tensor of shape (n, d)
  
  # Compute pairwise squared Euclidean distances
  C <- torch_cdist(x, y, p = 2)$pow(2)  # Shape: (n, n)
  return(C)
}

sinkhorn_loss <- function(x, y, epsilon, n, niter, p = 1) {
  # x: Tensor of shape (n, d)
  # y: Tensor of shape (n, d)
  # epsilon: Regularization parameter
  # n: Number of samples
  # niter: Number of iterations
  
  # Compute the cost matrix
  C <- torch_cdist(x, y, p = 2)$pow(p)  # Shape: (n, n)
  
  # Initialize uniform marginal distributions
  mu <- torch_full(c(n), 1.0 / n, dtype = torch_float(), requires_grad = FALSE)
  nu <- torch_full(c(n), 1.0 / n, dtype = torch_float(), requires_grad = FALSE)
  
  # Parameters
  rho <- 1.0    # Unbalanced transport parameter (can be adjusted)
  tau <- -0.8   # Nesterov-like acceleration parameter
  lam <- rho / (rho + epsilon)
  thresh <- 1e-1  # Stopping criterion
  
  # Define helper functions
  ave <- function(u, u1) {
    # Barycenter subroutine for kinetic acceleration
    tau * u + (1 - tau) * u1
  }
  
  M <- function(u, v) {
    # Modified cost for logarithmic updates
    # u: Tensor of shape (n)
    # v: Tensor of shape (n)
    # Returns a tensor of shape (n, n)
    (-C + u$unsqueeze(2) + v$unsqueeze(1)) / epsilon
  }
  
  lse <- function(A) {
    # Log-sum-exp function
    # A: Tensor of shape (n, n)
    # Returns a tensor of shape (n, 1)
    torch_logsumexp(A, dim = 2, keepdim = TRUE)
  }
  
  # Initialize dual variables
  u <- torch_zeros_like(mu)  # Shape: (n)
  v <- torch_zeros_like(nu)  # Shape: (n)
  err <- torch_tensor(0.0, dtype = torch_float())
  
  # Sinkhorn iterations
  for (i in 1:niter) {
    u1 <- u$clone()  # Save previous u
    
    # Compute M(u, v)
    M_uv <- M(u, v)  # Shape: (n, n)
    
    # Update u
    u <- epsilon * (torch_log(mu) - lse(M_uv)$squeeze()) + u  # Shape: (n)
    
    # Update v
    v <- epsilon * (torch_log(nu) - lse(M_uv$transpose(1, 2))$squeeze()) + v  # Shape: (n)
    
    # Compute error
    err <- torch_sum(torch_abs(u - u1))
    
    # Check stopping criterion
    if (err$item() < thresh) {
      break
    }
  }
  
  # Compute the transport plan
  pi <- torch_exp(M(u, v))  # Shape: (n, n)
  
  # Compute the Sinkhorn cost
  cost <- torch_sum(pi * C)
  
  return(cost)
}

sinkhorn_normalized <- function(x, y, epsilon, n, niter) {
  Wxy <- sinkhorn_loss(x, y, epsilon, n, niter)
  Wxx <- sinkhorn_loss(x, x, epsilon, n, niter)
  Wyy <- sinkhorn_loss(y, y, epsilon, n, niter)
  return(2 * Wxy - Wxx - Wyy)
}


# Sample data
set.seed(123)
n_samples <- 100
n_features <- 5
n_iter_sinkhorn <- 50
epsilon <- 0.1

# Create tensors for x and y
x_data <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)
y_data <- matrix(rnorm(n_samples * n_features), nrow = n_samples, ncol = n_features)

x <- torch_tensor(x_data)
y <- torch_tensor(y_data)

# Initialize transformation matrix with requires_grad = TRUE
transform <- torch_eye(n_features, n_features, requires_grad = TRUE)

# Define optimizer
optimizer <- optim_adam(params = list(transform), lr = 0.01)

# Training loop
num_epochs <- 100
for (epoch in 1:num_epochs) {
  optimizer$zero_grad()
  
  # Apply transformation to x
  x_transformed <- x$matmul(transform)
  
  # Compute Sinkhorn loss
  loss <- sinkhorn_loss(x_transformed, y, epsilon, n_samples, n_iter_sinkhorn)
  
  # Backpropagation
  loss$backward()
  
  # Update parameters
  optimizer$step()
  
  # Print loss every 10 epochs
  if (epoch %% 10 == 0) {
    cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
  }
}


## Harder

library(MASS)    # For mvrnorm function
library(torch)
library(ggplot2) # For visualization

# Set seed for reproducibility
set.seed(42)

# Number of samples and features
n_samples <- 200
n_features <- 2  # Using 2 features for visualization

# Dataset X
mean_X <- c(-2, -2)
cov_X <- matrix(c(1, 0.8, 0.8, 1), nrow = n_features)
data_X <- mvrnorm(n = n_samples, mu = mean_X, Sigma = cov_X)
X <- torch_tensor(data_X)

# Dataset Y
mean_Y <- c(3, 3)
cov_Y <- matrix(c(2, -0.5, -0.5, 1), nrow = n_features)
data_Y <- mvrnorm(n = n_samples, mu = mean_Y, Sigma = cov_Y)
Y <- torch_tensor(data_Y)

# Visualize initial datasets
df_initial <- data.frame(
  x1 = c(data_X[, 1], data_Y[, 1]),
  x2 = c(data_X[, 2], data_Y[, 2]),
  Dataset = factor(c(rep("X", n_samples), rep("Y", n_samples)))
)

ggplot(df_initial, aes(x = x1, y = x2, color = Dataset)) +
  geom_point(alpha = 0.6) +
  labs(title = "Initial Datasets X and Y",
       x = "Feature 1",
       y = "Feature 2") +
  theme_minimal()

# Sinkhorn distance function
sinkhorn_distance <- function(x, y, epsilon = 0.1, niter = 100, thresh = 1e-9) {
  n <- x$size(1)
  
  # Compute the cost matrix
  C <- torch_cdist(x, y, p = 2)
  
  # Marginal distributions
  mu <- torch_full(c(n, 1), 1.0 / n, dtype = torch_float(), requires_grad = FALSE)
  nu <- torch_full(c(n, 1), 1.0 / n, dtype = torch_float(), requires_grad = FALSE)
  
  # Initialize dual variables
  u <- torch_zeros_like(mu)
  v <- torch_zeros_like(nu)
  
  # Define helper functions
  M <- function(u, v) {
    (-C + u + v$t()) / epsilon
  }
  
  lse <- function(A) {
    torch_logsumexp(A, dim = 2, keepdim = TRUE)
  }
  
  # Sinkhorn iterations
  for (i in 1:niter) {
    u_prev <- u$clone()
    
    u <- epsilon * (torch_log(mu) - lse(M(u, v))) + u
    v <- epsilon * (torch_log(nu) - lse(M(u, v)$t())) + v
    
    err <- torch_norm(u - u_prev, p = 1)
    if (err$item() < thresh) {
      break
    }
  }
  
  # Compute transport plan
  pi <- torch_exp(M(u, v))
  
  # Compute Sinkhorn distance
  cost <- torch_sum(pi * C)
  
  return(cost)
}

# Initialize transformation parameters
transform <- torch_eye(n_features, n_features, requires_grad = TRUE)
translation <- torch_zeros(n_features, requires_grad = TRUE)

# Loss function
loss_function <- function(X, Y) {
  X_transformed <- X$matmul(transform) + translation
  loss <- sinkhorn_distance(X_transformed, Y, epsilon = 0.1, niter = 100)
  return(list(loss = loss, X_transformed = X_transformed))
}

# Optimizer
optimizer <- optim_adam(list(transform, translation), lr = 0.01)

# Training loop
loss_values <- c()
transformed_data_list <- list()

num_epochs <- 200
for (epoch in 1:num_epochs) {
  optimizer$zero_grad()
  
  res <- loss_function(X, Y)
  loss <- res$loss
  X_transformed <- res$X_transformed
  
  loss$backward()
  optimizer$step()
  
  loss_values <- c(loss_values, loss$item())
  
  if (epoch %% 20 == 0) {
    transformed_data_list[[length(transformed_data_list) + 1]] <- as.matrix(X_transformed$detach()$cpu())
    cat("Epoch:", epoch, "Loss:", loss$item(), "\n")
  }
}

# Plot the loss over epochs
df_loss <- data.frame(Epoch = 1:num_epochs, Loss = loss_values)
ggplot(df_loss, aes(x = Epoch, y = Loss)) +
  geom_line() +
  labs(title = "Sinkhorn Loss over Epochs",
       x = "Epoch",
       y = "Loss") +
  theme_minimal()

# Plot transformed data
data_transformed_initial <- transformed_data_list[[1]]
data_transformed_final <- transformed_data_list[[length(transformed_data_list)]]

df_transformed_initial <- data.frame(
  x1 = data_transformed_initial[, 1],
  x2 = data_transformed_initial[, 2],
  Dataset = "X_transformed_initial"
)

df_transformed_final <- data.frame(
  x1 = data_transformed_final[, 1],
  x2 = data_transformed_final[, 2],
  Dataset = "X_transformed_final"
)

df_plot_transformed <- rbind(
  data.frame(x1 = data_X[, 1], x2 = data_X[, 2], Dataset = "X_original"),
  data.frame(x1 = data_Y[, 1], x2 = data_Y[, 2], Dataset = "Y"),
  df_transformed_initial,
  df_transformed_final
)

ggplot(df_plot_transformed, aes(x = x1, y = x2, color = Dataset)) +
  geom_point(alpha = 0.6) +
  labs(title = "Datasets Before and After Transformation",
       x = "Feature 1",
       y = "Feature 2") +
  theme_minimal()
