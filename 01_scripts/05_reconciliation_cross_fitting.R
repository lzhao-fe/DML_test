# 05 replicate the coefficient estimation and se calculation with n_fold > 1 with cross fitting


# DML with linear nuisance function

library(DoubleML)
library(mlr3)
library(mlr3learners)
library(ggplot2)





# Prepare the data
data = fetch_401k(return_type = "data.table")
features_base = c("age", "inc", "educ", "fsize", "marr", "twoearn", "db", "pira", "hown")
data_dml_base = DoubleMLData$new(data, y_col = "net_tfa", d_cols = "e401", x_cols = features_base)

# Define linear regression learners for nuisance functions
linear_regression = lrn("regr.lm")       # For g(X): E[Y | X]
linear_classification = lrn("classif.log_reg")  # For m(X): E[D | X]


n_fold <- 3

set.seed(123)
dml_linear <- DoubleMLPLR$new(
  data_dml_base,
  ml_l = linear_regression,    # Outcome nuisance function
  ml_m = linear_classification, # Treatment nuisance function
  n_folds = n_fold,                 # No folds (use entire data)
  apply_cross_fitting = TRUE  
)


dml_linear$fit()
dml_linear$summary()


private_ <- dml_linear$.__enclos_env__$private
dml_smpl <- dml_linear_private$get__smpls()

res_dml <- private_$nuisance_est(private_$get__smpls())

# try to replicate with mlr3
# Define the learners for outcome and treatment models - same as what we used for DML

linear_regression_2 = lrn("regr.lm")  # Outcome model: ml_l
linear_classification_2 = lrn("classif.log_reg")  # Treatment model: ml_g

# note test and train prediction will be the same for n_fold = 1 and apply_cross_fitting = FALSE
linear_regression_2$predict_sets <- c("test", "train")
linear_classification_2$predict_type <- "prob"


# train-test split with no clustering, no repetition, only k-fold splitting

# n_fold > 1 so need to get train & test data separately
#https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml.R#L1183
# and is not clustered data: private_$is_cluster_data = FALSE
# NULL for n_rep
# private_$draw_sample_splitting_ = TRUE

#same page on double_ml, L486 -> split_samples function

n_rep <- 1

set.seed(123)
dummy_task = Task$new("dummy_resampling", 
                      "regr", 
                      data_dml_base$data)

dummy_resampling_scheme = rsmp("repeated_cv",
                               folds = n_fold,
                               repeats = n_rep)$instantiate(dummy_task)

train_ids = lapply(
  1:(n_fold * n_rep),
  function(x) dummy_resampling_scheme$train_set(x))


test_ids = lapply(
  1:(n_fold * n_rep),
  function(x) dummy_resampling_scheme$test_set(x))


smpls = lapply(1:n_rep, function(i_repeat) {
  list(
    train_ids = train_ids[((i_repeat - 1) * n_fold + 1):
                            (i_repeat * n_fold)],
    test_ids = test_ids[((i_repeat - 1) * n_fold + 1):
                          (i_repeat * n_fold)])
})


train_ids <- smpls[[1]][['train_ids']]
test_ids <- smpls[[1]][['test_ids']]
# train length(smpls[[1]][['train_ids']][[1]]) = 6610


# first estimate for each fold
# see specific models, e.g., PLR
# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/double_ml_plr.R#L382

# go back to dml_cv_predict
# https://github.com/DoubleML/doubleml-for-r/blob/c7edeb2c250eb1833d554d8fb58ab29a8b268781/R/helper.R
# L41 fold specific estimate


# task specification

n_obs <- nrow(data_dml_base$data)

# regression task

# --------------------------------------------

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

# gather predictions

l_hat <- rep(NA_real_, n_obs)
#if (testR6(obj_resampling, classes = "ResampleResult")) {
resampling_pred_regr = list(resampling_pred_regr)
#}
n_obj_rsmp_regr = length(resampling_pred_regr)
for (i_obj_rsmp in 1:n_obj_rsmp_regr) {
  f_hat = as.data.table(resampling_pred_regr[[i_obj_rsmp]]$prediction("test"))
  l_hat[f_hat[['row_ids']]] = f_hat[['response']]
}

l_hat[1:10]

identical(l_hat, res_dml$preds$ml_l)


# classification task

# --------------------------------------------

task_cls = TaskClassif$new(
  id = "ml_m", 
  backend = data %>% 
    mutate(e401 = factor(e401)) %>%
    select(-net_tfa), 
  target = "e401",
  positive = "1"
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



# gather predictions

m_hat <- rep(NA_real_, n_obs)
#if (testR6(obj_resampling, classes = "ResampleResult")) {
resampling_pred_cls = list(resampling_pred_cls)
#}
n_obj_rsmp_cls = length(resampling_pred_cls)
for (i_obj_rsmp in 1:n_obj_rsmp_cls) {
  f_hat = as.data.table(resampling_pred_cls[[i_obj_rsmp]]$prediction("test"))
  m_hat[f_hat[['row_ids']]] = f_hat[['prob.1']]
}

m_hat[1:10]

identical(m_hat, res_dml$preds$ml_m)





# Compute stats

d <- data_dml_base$data$e401
y <- data_dml_base$data$net_tfa


# score_elements function

v_hat  <-  d - m_hat
u_hat  <-  y - l_hat
v_hatd <-  v_hat * d


psi_a <- -v_hat * v_hat
psi_b <- v_hat * u_hat


theta = -mean(psi_b) / mean(psi_a)

print(paste0('Treatment effect estimate: ', theta))


# ///////////

psi <- psi_a * theta + psi_b

# note that when applying cross fitting, the scaling factor is n_obs, otherwise = test sample size
var_scaling_factor <- n_obs 

J <- mean(psi_a)

sigma2_hat <- mean(psi^2) / (J^2) / var_scaling_factor

std_err <- sqrt(sigma2_hat)

print(paste0('Standard error: ', std_err))
