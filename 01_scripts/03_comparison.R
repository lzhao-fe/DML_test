# 03 linear learner and OLS comparison

# OLS and linear learner in mlr3 are giving the same result

# -------------------------------------------------------


# Load necessary library
library(stats)# Load necessary libraries
library(data.table)

# Example data
data <- data.table(fetch_401k(return_type = "data.table"))


ols_model <- lm(net_tfa ~ e401 + ., data = data)
summary(ols_model)




library(mlr3)
library(mlr3learners)

# define task
task <- TaskRegr$new(id = "regr_task", backend = data, target = "net_tfa")

# define learner
linear_regression = lrn("regr.lm")


# Train the model
linear_regression$train(task)


# Access the fitted model
lr_model <- linear_regression$model

# Get the coefficients
coefficients <- lr_model$coefficients



# Linear learner

# ------------------------------------

# DML


# DML with linear nuisance function

library(DoubleML)
library(mlr3)
library(mlr3learners)

# Prepare the data
data = fetch_401k(return_type = "data.table")
features_base = c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
data_dml_base = DoubleMLData$new(data, y_col = "net_tfa", d_cols = "e401", x_cols = features_base)

# Define linear regression learners for nuisance functions
linear_regression = lrn("regr.lm")       # For g(X): E[Y | X]
linear_classification = lrn("classif.log_reg")  # For m(X): E[D | X]

# Initialize DML-PLR model without cross-fitting and with n_folds = 1
dml_plr_linear = DoubleMLPLR$new(
  data_dml_base,
  ml_l = linear_regression,    # Outcome nuisance function
  ml_m = linear_classification, # Treatment nuisance function
  n_folds = 1,                 # No folds (use entire data)
  apply_cross_fitting = FALSE  # Disable cross-fitting
)

# Fit the model
dml_plr_linear$fit()

# Display results
dml_plr_linear$summary()




# try to replicate

# Define task for outcome and treatment
task_outcome <- TaskRegr$new(
  id = "outcome_task", 
  backend = data %>% 
    select(-e401), 
  target = "net_tfa"
)

task_treatment <- TaskClassif$new(
  id = "treatment_task", 
  backend = data %>% 
    select(-net_tfa) %>%
    mutate(e401 = factor(e401)), 
  target = "e401")



# Define the learners for outcome and treatment models
linear_regression_2 = lrn("regr.lm")  # Outcome model: g(X)
linear_classification_2 = lrn("classif.log_reg")  # Treatment model: m(X)


# Step 1: Fit outcome model (g(X)) using linear regression
linear_regression_2$train(task_outcome)
# g_hat <- linear_regression_2$predict(task_outcome)$score()  # Predicted outcome


# Step 2: Fit treatment model (m(X)) using logistic regression
linear_classification_2$train(task_treatment)
# m_hat <- linear_classification_2$predict(task_treatment)$score()  # Predicted treatment

# 
# 
# # Step 3: Calculate residuals
# residual_y <- data$net_tfa - g_hat  # Residuals for outcome
# residual_d <- data$e401 - m_hat    # Residuals for treatment
# 
# 
# # Step 4: Regress residuals of outcome on residuals of treatment
# orthogonalized_model <- lm(residual_y ~ residual_d - 1)  # No intercept
# summary(orthogonalized_model)  # Treatment effect estimate
# 
# 
# 
# # Step 5: Variance estimation (basic estimate)
# # Use standard error from the OLS regression to estimate variance
# se <- summary(orthogonalized_model)$coefficients[, 2]
# t_value <- summary(orthogonalized_model)$coefficients[, 3]
# p_value <- summary(orthogonalized_model)$coefficients[, 4]
# 
# 
# # Print results for treatment effect
# cat("Treatment effect estimate:", orthogonalized_model$coefficients, "\n")
# cat("Standard error:", se, "\n")
# cat("t-value:", t_value, "\n")
# cat("p-value:", p_value, "\n")
# 
# # Step 6: Confidence interval for treatment effect
# confint(orthogonalized_model)




# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml.R#L400
res_dml_res_lz <- dml_plr_linear$.__enclos_env__$private$nuisance_est(dml_plr_linear$.__enclos_env__$private$get__smpls())



# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml.R#L380


# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml_plr.R#L382

# nuisance_est

# l_hat_lz <- linear_regression_2$predict(task_outcome)$response
# m_hat_lz <- as.integer(linear_classification_2$predict(task_treatment)$response) - 1


l_hat_lz <- res_dml_res_lz$preds$ml_l
m_hat_lz <- res_dml_res_lz$preds$ml_m


d_lz <- data$e401
y_lz <- data$net_tfa

# if ml_g exists
#psi_a_lz <- -(d_lz - m_hat_lz) * (d_lz - m_hat_lz)
#psi_b_lz <-  (d_lz - m_hat_lz) * (y_lz - l_hat_lz)


# dml_plr_linear$score == 'partialling out'

## https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml_plr.R#L444

# score_elements function

v_hat_lz  <-  d_lz - m_hat_lz
u_hat_lz  <-  y_lz - l_hat_lz
v_hatd_lz <-  v_hat_lz * d_lz


psi_a_lz <- -v_hat_lz * v_hat_lz
psi_b_lz <- v_hat_lz * u_hat_lz


# double_ml L414 compute_score


## https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml_plr.R#L1647
# modified from the original script

orth_est <- function(psi_a, psi_b, 
                     inds = NULL) {

  if (!is.null(inds)) {
    psi_a = psi_a[inds]
    psi_b = psi_b[inds]
  }
  theta = -mean(psi_b) / mean(psi_a)
  return(theta)
}


#dml_plr_linear$dml_procedure == 'dml1'
# to replicate dml
dml_smpls_lz <- dml_plr_linear$.__enclos_env__$private$get__smpls()
test_ids_lz <- dml_smpls_lz$test_ids

dml_private <- dml_plr_linear$.__enclos_env__$private


thetas = rep(NA_real_, length(test_ids_lz))

for (i_fold in seq_len(length(test_ids_lz))) {
  test_index = test_ids_lz[[i_fold]]
  thetas[i_fold] = orth_est(psi_a_lz, psi_b_lz,
                            inds = test_index)
}
coef_lz = mean(thetas, na.rm = TRUE)


# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml.R#L1508
# se_causal_pars 
# -> L1562 var_est


# L1705 compute_score
# private$get__psi_a() * private$get__all_coef() + private$get__psi_b()

psi_lz <- psi_a_lz * coef_lz + psi_b_lz


# back to var_est L1562

# dml_plr_linear$apply_cross_fitting = FALSE
# 
var_scaling_factor_lz <- length(test_ids_lz[[1]])

J_lz <- mean(psi_a_lz)
sigma2_hat_lz <- mean(psi_lz^2) / (J_lz^2) / var_scaling_factor_lz
sqrt(sigma2_hat_lz)




#https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/helper.R

# L265 initiate_task

# valid_task_type = c("regr", "classif")

# initiate_learner

# regression

task_regr_lz = TaskRegr$new(
  id = 'ml_l', 
  backend = data %>%
    select(-e401), 
  target = "net_tfa")


#https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/helper.R#L259
linear_regression_2$predict_sets <- c("test", "train")



# ml_learner = initiate_learner(
#   learner, task_type,
#   est_params, return_train_preds)
# resampling_smpls = rsmp("custom")$instantiate(
#   task_pred, smpls$train_ids,
#   smpls$test_ids)
# resampling_pred = resample(task_pred, ml_learner, resampling_smpls,
#                            store_models = TRUE)
# preds = extract_prediction(resampling_pred, task_type, n_obs)
# models = extract_models(resampling_pred)
# if (return_train_preds) {
#   train_preds = extract_prediction(resampling_pred, task_type, n_obs,
#                                    return_train_preds = TRUE)

resampling_smpls_lz <- rsmp("custom")$instantiate(
  task_regr_lz, 
  dml_smpls_lz$train_ids,
  dml_smpls_lz$test_ids)

resampling_pred_lz <- resample(
  task_regr_lz, 
  linear_regression_2, 
  resampling_smpls_lz,
  store_models = TRUE)



# ///////////

# Initialize placeholders
train_predictions_list <- vector("list", resampling_pred_lz$resampling$iters)

# Loop through each fold
for (i in seq_len(resampling_pred_lz$resampling$iters)) {
  # Get the training and test indices
  train_indices <- resampling_pred_lz$resampling$train_set(i)
  test_indices <- resampling_pred_lz$resampling$test_set(i)
  
  # Extract training data for this fold
  train_data <- data[train_indices, ]
  test_data <- data[test_indices, ]
  
  # Train the model on training data
  linear_regression_2$train(TaskRegr$new(id = "fold_train", backend = train_data, target = "net_tfa"))
  
  # Predict on training data
  train_predictions <- linear_regression_2$predict(TaskRegr$new(id = "fold_train", backend = train_data, target = "net_tfa"))
  train_predictions_list[[i]] <- train_predictions
}

# ///////////



#https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/helper.R#L256
linear_classification_2$predict_type <- "prob"

task_cls_lz = TaskClassif$new(id = "ml_m", 
                              backend = data %>% mutate(e401 = factor(e401)) %>% select(-net_tfa), 
                              target = "e401",
                              positive = "1")


resampling_smpls_cls_lz <- rsmp("custom")$instantiate(
  task_cls_lz, 
  dml_smpls_lz$train_ids,
  dml_smpls_lz$test_ids)

resampling_pred_cls_lz <- resample(
  task_cls_lz, 
  linear_classification_2, 
  resampling_smpls_cls_lz,
  store_models = TRUE)


predicted_prob <- as.data.table( resampling_pred_cls_lz$prediction()) %>% select(`prob.1`) %>% unlist() %>% unname()

# L170 extract_prediction

n_iters_lz <- resampling_pred_lz$resampling$iters
preds_lz <-  vector("list", n_iters_lz)

# placeholders
resp_name <- "response"
ind_name <- 'id_' 


# Iterate through each fold to extract predictions
for (i_iter in 1:n_iters_lz) {
  preds_vec <- rep(NA_real_, n_obs)  # Placeholder for all predictions
  f_hat <- as.data.table(resampling_pred_lz$predictions(i_iter))  # Get predictions for fold
  
  # Verify column names for indices and responses
  ind_name <- "row_id"  # Check this column matches task indices
  resp_name <- "response"  # Verify column for predictions
  
  # Map predictions to correct indices
  preds_vec[f_hat[[ind_name]]] <- f_hat[[resp_name]]
  preds_lz[[i_iter]] <- preds_vec
}












f_hat_list_lz <- lapply(
  1:n_iters_lz,
  function(x) as.data.table(resampling_pred_lz$predictions("train")[[x]]))

for (i_iter in 1:n_iters_lz) {
  preds_vec = rep(NA_real_, n_obs)
  f_hat = f_hat_list_lz[[i_iter]]
  preds_vec[f_hat[[ind_name]]] = f_hat[[resp_name]]
  preds_lz[[i_iter]] = preds_vec
}



f_hat_lz <- resampling_pred_lz


preds_lz = extract_prediction(resampling_pred_lz, task_type, n_obs)
models_lz = extract_models(resampling_pred)

task_outcome <- TaskRegr$new(id = "outcome_task", backend = data, target = "net_tfa")
task_treatment <- TaskClassif$new(id = "treatment_task", backend = data %>% mutate(e401 = factor(e401)), target = "e401")


dml_cv_predict(linear_regression_2, X_cols = )