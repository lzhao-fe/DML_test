library(DoubleML)
library(mlr3)
library(mlr3learners)
library(data.table)
library(ggplot2)


# https://docs.doubleml.org/stable/examples/R_double_ml_pension.html


#lapply(c('DoubleML', 'mlr3', 'mlr3learners', 'data.table', 'ggplot2'), require, character.only = TRUE)
#install.packages(c('DoubleML', 'mlr3', 'mlr3learners', 'data.table', 'ggplot2', 
#                   'glmnet', 'ranger', 'rpart', 'xgboost'))

# suppress messages during fitting
lgr::get_logger("mlr3")$set_threshold("warn")

# load data as a data.table
data = fetch_401k(return_type = "data.table", instrument = TRUE)
dim(data)
str(data)




# /////

hist_e401 = ggplot(data, aes(x = e401, fill = factor(e401))) +
  geom_bar() + theme_minimal() +
  ggtitle("Eligibility, 401(k)") +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
hist_e401




# /////

hist_p401 = ggplot(data, aes(x = p401, fill = factor(p401))) +
  geom_bar() + theme_minimal() +
  ggtitle("Participation, 401(k)") +
  theme(legend.position = "bottom", plot.title = element_text(hjust = 0.5),
        text = element_text(size = 20))
hist_p401



# /////

dens_net_tfa = ggplot(data, aes(x = net_tfa, color = factor(e401), fill = factor(e401)) ) +
  geom_density() + xlim(c(-20000, 150000)) +
  facet_wrap(.~e401)  + theme_minimal() +
  theme(legend.position = "bottom", text = element_text(size = 20))

dens_net_tfa


# /////

APE_e401_uncond = data[e401==1, mean(net_tfa)] - data[e401==0, mean(net_tfa)]
round(APE_e401_uncond, 2)



# /////

APE_p401_uncond = data[p401==1, mean(net_tfa)] - data[p401==0, mean(net_tfa)]
round(APE_p401_uncond, 2)


# /////


# Set up basic model: Specify variables for data-backend
features_base = c("age", "inc", "educ", "fsize",
                  "marr", "twoearn", "db", "pira", "hown")

# Initialize DoubleMLData (data-backend of DoubleML)
data_dml_base = DoubleMLData$new(data,
                                 y_col = "net_tfa",
                                 d_cols = "e401",
                                 x_cols = features_base)
data_dml_base



# /////

# Set up a model according to regression formula with polynomials
formula_flex = formula(" ~ -1 + poly(age, 2, raw=TRUE) +
                        poly(inc, 2, raw=TRUE) + poly(educ, 2, raw=TRUE) +
                        poly(fsize, 2, raw=TRUE) + marr + twoearn +
                        db + pira + hown")
features_flex = data.frame(model.matrix(formula_flex, data))

model_data = data.table("net_tfa" = data[, net_tfa],
                        "e401" = data[, e401],
                        features_flex)

# Initialize DoubleMLData (data-backend of DoubleML)
data_dml_flex = DoubleMLData$new(model_data,
                                 y_col = "net_tfa",
                                 d_cols = "e401")

data_dml_flex



# /////

# Partially Linear Regression Model (PLR)
set.seed(123)
lasso = lrn("regr.cv_glmnet", nfolds = 5, s = "lambda.min")
lasso_class = lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min")

# Initialize DoubleMLPLR model
dml_plr_lasso_base = DoubleMLPLR$new(data_dml_base,
                                ml_l = lasso,
                                ml_m = lasso_class,
                                n_folds = 3)
dml_plr_lasso_base$fit()
dml_plr_lasso_base$summary()




# /////


# Initialize learners
set.seed(123)
lasso = lrn("regr.cv_glmnet", nfolds = 5, s = "lambda.min")
lasso_class = lrn("classif.cv_glmnet", nfolds = 5, s = "lambda.min")

# Initialize DoubleMLPLR model
dml_plr_lasso_flex = DoubleMLPLR$new(data_dml_flex,
                                ml_l = lasso,
                                ml_m = lasso_class,
                                n_folds = 3)
dml_plr_lasso_flex$fit()
dml_plr_lasso_flex$summary()


# /////


# Random Forest
randomForest = lrn("regr.ranger", max.depth = 7,
                   mtry = 3, min.node.size = 3)
randomForest_class = lrn("classif.ranger", max.depth = 5,
                         mtry = 4, min.node.size = 7)

set.seed(123)
dml_plr_forest = DoubleMLPLR$new(data_dml_base,
                                 ml_l = randomForest,
                                 ml_m = randomForest_class,
                                 n_folds = 3)
dml_plr_forest$fit()
dml_plr_forest$summary()



# /////


# Trees
trees = lrn("regr.rpart", cp = 0.0047, minsplit = 203)
trees_class = lrn("classif.rpart", cp = 0.0042, minsplit = 104)

set.seed(123)
dml_plr_tree = DoubleMLPLR$new(data_dml_base,
                               ml_l = trees,
                               ml_m = trees_class,
                               n_folds = 3)
dml_plr_tree$fit()
dml_plr_tree$summary()


# /////


# Boosted trees
boost = lrn("regr.xgboost",
            objective = "reg:squarederror",
            eta = 0.1, nrounds = 35)
boost_class = lrn("classif.xgboost",
                  objective = "binary:logistic", eval_metric = "logloss",
                  eta = 0.1, nrounds = 34)

set.seed(123)
dml_plr_boost = DoubleMLPLR$new(data_dml_base,
                                ml_l = boost,
                                ml_m = boost_class,
                                n_folds = 3)
dml_plr_boost$fit()
dml_plr_boost$summary()



# //////



# Load necessary library
library(stats)

# Assuming `data` is a dataframe containing Y, D, and X (covariates)
# X is a matrix or data frame of covariates

ols_data <- data %>%
  select()

# Fit OLS regression
ols_model <- lm(net_tfa ~ e401 + ., data = data_dml_base$data_model)

# View summary
model_summary <- summary(ols_model)

ols_confint <- confint(ols_model)

required_row_name <- 'e401'
ols_d_confint <- matrix(ols_confint[which(rownames(ols_confint) == required_row_name), ], nrow = 1)
rownames(ols_d_confint) <- required_row_name
colnames(ols_d_confint) <- c("2.5 %",  "97.5 %")

ols_coefficient <-  ols_model$coefficients[which(names( ols_model$coefficients) == required_row_name)]
ols_se <- model_summary$coefficients[which(rownames(model_summary$coefficients) == required_row_name), 
                                     which(colnames(model_summary$coefficients)== 'Std. Error')]

# /////

confints = rbind(ols_d_confint,
                 dml_plr_lasso_base$confint(), dml_plr_lasso_flex$confint(),
                 dml_plr_forest$confint(),
                 dml_plr_tree$confint(), dml_plr_boost$confint())

estimates = c(ols_coefficient,
              dml_plr_lasso_base$coef, dml_plr_lasso_flex$coef,
              dml_plr_forest$coef,
              dml_plr_tree$coef, dml_plr_boost$coef)

std_err <- c(ols_se,
             dml_plr_lasso_base$se, dml_plr_lasso_flex$se,
             dml_plr_forest$se,
             dml_plr_tree$se, dml_plr_boost$se)

result_plr = data.table("model" = c('OLS', "PLR", "PLR", 
                                    "PLR", "PLR", "PLR"),
                        "ML" = c('1. OLS',
                                 "2. glmnet_base", "3. glmnet_flex", 
                                 "4. ranger", "5. rpart", "6. xgboost"),
                        "Estimate" = estimates,
                        'Std. Error' = std_err,
                        "lower" = confints[,1],
                        "upper" = confints[,2])
result_plr


# /////

g_ci = ggplot(result_plr, aes(x = ML, y = Estimate, color = ML)) +
  geom_point() +
  geom_errorbar(aes(ymin = lower, ymax = upper, color = ML))  +
  geom_hline(yintercept = 0, color = "grey") +
  theme_minimal() + ylab("Coefficients and 0.95- confidence interval") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none",
        text = element_text(size = 20))

g_ci






