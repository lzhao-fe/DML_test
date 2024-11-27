# 06 2SLS and DML



# Load necessary libraries
library(DoubleML)
library(mlr3)
library(mlr3learners)


data <- fetch_401k(return_type = "data.table")
features_base <- c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
outcome_var <- 'net_tfa'
treatment_var <- 'e401'





# Create the DoubleML dataset
data_dml_base = DoubleMLData$new(
  data, 
  y_col = outcome_var, 
  d_cols = treatment_var, 
  x_cols = features_base
)


# Define linear regression learners for nuisance functions
linear_regression <- lrn("regr.lm")       
linear_classification <- lrn("regr.lm")  


# Set up the DML-PLR model
dml_plr_linear <- DoubleMLPLR$new(
  data_dml_base,
  ml_l = linear_regression,    # Outcome nuisance function
  ml_m = linear_classification, # Treatment nuisance function
  n_folds = 1,                 # Disable cross-fitting
  apply_cross_fitting = FALSE
)

# Fit the model
dml_plr_linear$fit()

# Display results
dml_plr_linear$summary()



# /////


linear_regression_2 = lrn("regr.lm")  # Outcome model: ml_l
linear_classification_2 = lrn("regr.lm")  # Treatment model: ml_g

train_ids <- list(seq(1, dim(data)[1]))
test_ids <- list(seq(1, dim(data)[1]))


task_regr = TaskRegr$new(
  id = 'ml_l', 
  backend = data %>%
    select(-e401),  # remember to remove the treatment variable
  target = "net_tfa"
)


resampling_smpls_regr <- rsmp("custom")$instantiate(
  task_regr, 
  train_ids,
  test_ids
)

resampling_pred_regr <- resample(
  task_regr, 
  linear_regression_2, 
  resampling_smpls_regr,
  store_models = TRUE
)



# classification

task_cls = TaskRegr$new(
  id = "ml_m", 
  backend = data %>% 
    #mutate(e401 = factor(e401)) %>%
    select(-net_tfa), 
  target = "e401"
)


resampling_smpls_cls <- rsmp("custom")$instantiate(
  task_cls, 
  train_ids,
  test_ids
)

resampling_pred_cls <- resample(
  task_cls, 
  linear_classification_2, 
  resampling_smpls_cls,
  store_models = TRUE
)

# ---------------
l_hat <- resampling_pred_regr$prediction()$response
m_hat <- resampling_pred_cls$prediction()$response
d <- data$e401
y <- data$net_tfa

# score_elements function

v_hat  <-  d - m_hat
u_hat  <-  y - l_hat
v_hatd <-  v_hat * d


psi_a <- -v_hat * v_hat
psi_b <- v_hat * u_hat


theta = -mean(psi_b) / mean(psi_a)

print(paste0('Treatment effect estimate: ', theta))


psi <- psi_a * theta + psi_b

var_scaling_factor <- length(test_ids[[1]])

J <- mean(psi_a)

sigma2_hat <- mean(psi^2) / (J^2) / var_scaling_factor

std_err <- sqrt(sigma2_hat)

print(paste0('Standard error: ', std_err))



# //////








# Step 1: Residualize Y (outcome)
outcome_formula <- as.formula(paste(outcome_var, "~", paste(features_base, collapse = "+")))
outcome_model <- lm(outcome_formula, data = data)
residual_y <- residuals(outcome_model)

# Step 2: Residualize D (treatment)
treatment_formula <- as.formula(paste(treatment_var, "~", paste(features_base, collapse = "+")))
treatment_model <- lm(treatment_formula, data = data)
residual_d <- residuals(treatment_model)

# Step 3: Regress residualized Y on residualized D
orthogonal_ols <- lm(residual_y ~ residual_d - 1)  # No intercept
summary(orthogonal_ols)




# Extract residuals from the second stage regression
residuals_2nd_stage <- residuals(orthogonal_ols)

# Calculate the variance of the residuals
sigma_squared <- sum(residuals_2nd_stage^2) / (length(residuals_2nd_stage) - 1)

# Calculate the standard error of the coefficient
X <- model.matrix(orthogonal_ols)
XtX_inv <- solve(t(X) %*% X)
se_beta <- sqrt(sigma_squared * XtX_inv[1, 1])

# Print the standard error
se_beta