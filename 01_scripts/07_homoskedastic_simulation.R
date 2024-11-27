# set.seed(123)  # For reproducibility
# n <- 100000
# 
# # Generate independent variables
# X <- rnorm(n, mean = 5, sd = 2)  # Variable of interest
# D <- rnorm(n, mean = 10, sd = 3)  # Confounding variable
# 
# Y <- rnorm(n, mean = 3, sd = 1) 
# 
# # One step
# one_step_model <- lm(Y ~ X + D)
# 
# # Two step
# resid_YD <- lm(Y ~ D)$residuals
# resid_XD <- lm(X ~ D)$residuals
# two_step_model <- lm(resid_YD ~ resid_XD)
# 
# # Check std error on X in model 1 vs std error on resid_XD in model 2
# summary(one_step_model)
# summary(two_step_model)


# -----------------

# DML with no cross-fitting and linear models
library(DoubleML)
library(mlr3)
library(mlr3learners)

set.seed(123) # For reproducibility

# Generate data
n <- 5000000  # Sample size
p <- 5     # Number of covariates

# Covariates (features)
X <- matrix(rnorm(n * p), n, p)
colnames(X) <- paste0("X", 1:p)

# Coefficients
beta <- rnorm(p)         # Coefficients for covariates
gamma <- rnorm(p)        # Coefficients for treatment model
theta <- 2               # True treatment effect

# Treatment and Outcome
D <- X %*% gamma + rnorm(n)  # Treatment
Y <- D * theta + X %*% beta + rnorm(n)  # Outcome

# Combine into a data frame
data <- data.frame(Y = Y, D = D, X)

# # OLS Regression
# ols_model <- lm(Y ~ D + ., data = data)
# ols_summary <- summary(ols_model)


ols_linear <- lrn('regr.lm')
task_ols <- TaskRegr$new(
  id = 'ols', 
  backend = data,  # remember to remove the treatment variable
  target = "Y"
)

# Train the model
ols_linear$train(task_ols)


# Extract the fitted model (lm object)
fitted_model <- ols_linear$model

# Summarize the fitted model to get standard errors
model_summary <- summary(fitted_model)

# Extract standard error for the treatment coefficient (D)
se_treatment <- model_summary$coefficients["D", "Std. Error"]

# Print results
cat("Treatment Effect Standard Error (OLS):", se_treatment, "\n")

# If you want all coefficients and their SEs
#print(model_summary$coefficients)





#ols_linear


# Define task
dml_data <- DoubleMLData$new(data, y_col = "Y", d_cols = "D", x_cols = colnames(X))

# Define linear learners for DML
linear_regression <- lrn("regr.lm")       
linear_classification <- lrn("regr.lm")  



# DML Model
dml_model <- DoubleMLPLR$new(dml_data, ml_l = linear_regression, ml_m = linear_classification, n_fold = 1, apply_cross_fitting = FALSE)
dml_model$fit()
dml_summary <- dml_model$summary()



# Results
paste0("OLS Estimate for Theta: ", fitted_model$coefficients['D'])#ols_summary$coefficients["D", "Estimate"])
paste0("DML Estimate for Theta: ", dml_model$coef)
paste0("OLS Standard Error for Theta: ", se_treatment)#ols_summary$coefficients["D", "Std. Error"])
paste0("DML Standard Error for Theta: ", dml_model$se)








