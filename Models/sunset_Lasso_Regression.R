##############################
### Lasso Regression Model ###
##############################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")

set.seed(8488)


# Setup Lasso model and workflow, tune penalty and mixture parameters
sunset_lasso_spec <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

sunset_lasso_workflow <- workflow() %>% 
  add_recipe(sunset_recipe) %>% 
  add_model(sunset_lasso_spec)


# creating parameter grid to tune ranges of hyper parameters
lasso_pen_mix_grid <- grid_regular(penalty(range = c(-5, 5)), mixture(range = c(0,1)), levels = 10)
lasso_pen_mix_grid


# fit the models to our folded data using `tune_grid()`.
sunset_lasso_tune_res <- tune_grid(
  sunset_lasso_workflow,
  resamples = sunset_folds, 
  grid = lasso_pen_mix_grid
)


# use `select_best()` to choose the model that has the optimal `roc_auc`.
collect_metrics(sunset_lasso_tune_res)
best_sunset_lasso_penalty <- select_best(sunset_lasso_tune_res, metric = "roc_auc")
best_sunset_lasso_penalty


# use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_lasso_final <- finalize_workflow(sunset_lasso_workflow, best_sunset_lasso_penalty)
sunset_lasso_final_fit <- fit(sunset_lasso_final, data = sunset_train)


# saving data to load into rmd file
save(lasso_tune_res, sunset_lasso_final_fit, 
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Lasso_Regression.rda")