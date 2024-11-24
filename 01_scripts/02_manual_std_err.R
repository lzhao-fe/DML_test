# Load libraries
library(data.table)
library(mlr3)
library(mlr3learners)
library(DoubleML)




# Load dataset
data = fetch_401k(return_type = "data.table", instrument = TRUE)

# Define covariates
features_base = c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")



# Fit the model
set.seed(123)


# applying package
# ------------------------------------


# Prepare data for DoubleML
data_dml_base = DoubleMLData$new(data,
                                 y_col = "net_tfa",
                                 d_cols = "e401",
                                 x_cols = features_base)



# Initialize LASSO learners
lasso = lrn("regr.cv_glmnet", nfolds = 5, s = "lambda.min")
lasso_class = lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min")



# Initialize DoubleMLPLR model

dml_plr_lasso_base_n_fold = DoubleMLPLR$new(data_dml_base, 
                                            ml_l = lasso, 
                                            ml_m = lasso_class, 
                                            n_folds = 5,
                                            apply_cross_fitting = TRUE) # to be able to compare without cross fitting (do not accept n_fold >2)

# Fit the model
dml_plr_lasso_base_n_fold$fit()

# Display results
dml_plr_lasso_base_n_fold$summary()




dml_plr_lasso_base_2_fold = DoubleMLPLR$new(data_dml_base, 
                                     ml_l = lasso, 
                                     ml_m = lasso_class, 
                                     n_folds = 2,
                                     apply_cross_fitting = TRUE) # to be able to compare without cross fitting (do not accept n_fold >2)

# Fit the model
dml_plr_lasso_base_2_fold$fit()

# Display results
dml_plr_lasso_base_2_fold$summary()



# Initialize DoubleMLPLR model
dml_plr_lasso_base_no_cfit = DoubleMLPLR$new(data_dml_base, 
                                             ml_l = lasso, 
                                             ml_m = lasso_class, 
                                             n_folds = 2, 
                                             apply_cross_fitting = FALSE)


dml_plr_lasso_base_no_cfit$fit()

# Display results
dml_plr_lasso_base_no_cfit$summary()



# manual
# ------------------------------------



psi_a <- dml_plr_lasso_base_n_fold$.__enclos_env__$private$get__psi_a()
psi <- dml_plr_lasso_base_n_fold$.__enclos_env__$private$get__psi()
n_obs <- dml_plr_lasso_base_n_fold$data$n_obs




# Apply cross-fitting check
if (dml_plr_lasso_base_n_fold$apply_cross_fitting) {
  scaling_factor <- n_obs
} else {
  # Without cross-fitting, extract test sample indices
  smpls <- dml_plr_lasso_base_n_fold$.__enclos_env__$private$get__smpls()
  test_ids <- smpls$test_ids
  test_index <- test_ids[[1]]  # Use the first fold for simplicity
  psi_a <- psi_a[test_index]
  psi <- psi[test_index]
  scaling_factor <- length(test_index)
}



# Compute the variance and standard error
J <- mean(psi_a)  # Average of psi_a
sigma2_hat <- mean(psi^2) / (J^2) / scaling_factor  # Variance estimate
std_error <- sqrt(sigma2_hat)  # Standard error




cat("Manually calculated standard error:", std_error, "\n")






# OLS


# Get the design matrix
X <- model.matrix(ols_model)

# Calculate residuals
residuals <- resid(ols_model)

# Estimate residual variance
residual_variance <- sum(residuals^2) / ols_model$df.residual

# Compute the variance-covariance matrix of the coefficients
vcov_matrix <- residual_variance * solve(t(X) %*% X)

# Extract standard errors
standard_errors <- sqrt(diag(vcov_matrix))

# Print the standard errors
print(standard_errors)

