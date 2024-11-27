# 06 2SLS and DML

# Define variables
outcome <- "Y"            # Outcome variable
treatment <- "D"          # Treatment variable
covariates <- c("X1", "X2", "X3")  # Covariates
data <- data.frame(Y = rnorm(100), D = rbinom(100, 1, 0.5), X1 = rnorm(100), X2 = rnorm(100), X3 = rnorm(100))

data <- fetch_401k(return_type = "data.table")
features_base <- c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
outcome_var <- 'net_tfa'
treatment_var <- 'e401'



# Load necessary libraries
library(DoubleML)
library(mlr3)
library(mlr3learners)

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




# -------------
# Extract the stored models from the resampling result

# Loop through each model to extract coefficients and standard errors
for (i in seq_along(models)) {
  regr_model <- resampling_pred_regr$learners[[i]]$model
  regr_summary <- summary(regr_model)
  
  # Store coefficients and standard errors
  #coefficients_list[[i]] <- summary_model$coefficients[, "Estimate"]
  #se_list[[i]] <- summary_model$coefficients[, "Std. Error"]
}



outcome_model$coefficients - regr_summary$coefficients[, 'Estimate'][c('(Intercept)' , features_base)]



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



# Loop through each model to extract coefficients and standard errors
for (i in seq_along(models)) {
  cls_model <- resampling_pred_cls$learners[[i]]$model
  cls_summary <- summary(cls_model)
  
  # Store coefficients and standard errors
  #coefficients_list[[i]] <- summary_model$coefficients[, "Estimate"]
  #se_list[[i]] <- summary_model$coefficients[, "Std. Error"]
}

cls_summary$coefficients[, 'Estimate'][c('(Intercept)' , features_base)]

# ---------------
l_hat <- resampling_pred_regr$prediction()$response
m_hat <- resampling_pred_cls$prediction()$response
d <- data$e401
y <- data$net_tfa

# score_elements function

v_hat  <-  d - m_hat
u_hat  <-  y - l_hat
#v_hatd <-  v_hat * d


psi_a <- -v_hat * v_hat
psi_b <- v_hat * u_hat


theta = -mean(psi_b) / mean(psi_a)

print(paste0('Treatment effect estimate: ', theta))


psi <- psi_a * theta + psi_b #  psi = (psi_b -v_hat * v_hat * theta) = psi_a * theta + psi_b

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




# manually replicate the standard error for 2SLS

residuals_orthogonal <- residuals(orthogonal_ols)
sigma_squared <- sum(residuals_orthogonal^2) / (length(residuals_orthogonal) - 1)


# Compute the Variance of Residualized D
var_residual_d <- var(residual_d)

# 
n <- nrow(data)
var_beta <- sigma_squared / (n * var_residual_d)

#

se_beta <- sqrt(var_beta)

# residual_d ~ v_hat
sum(residual_d - v_hat)


# confidence interval

coef_orthogonal <- coef(orthogonal_ols)["residual_d"]
critical_value <- qt(0.975, df = n - 1)
ci_lower <- coef_orthogonal - critical_value * se_beta
ci_upper <- coef_orthogonal + critical_value * se_beta
