###########################
### Random Forest Model ###
###########################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")

set.seed(8488)


# Setup a random forest model and workflow. Use the `ranger` engine and set 
#`importance = "impurity"`. Let's also tune `mtry`, `trees`, and `min_n`.
sunset_rand_forest_spec <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

sunset_rand_forest_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(sunset_rand_forest_spec)


# creating parameter grid to tune ranges of hyper parameters
sunset_rf_param_grid <- grid_regular(mtry(range = c(2, 15)), trees(range = c(2, 10)), 
                                  min_n(range = c(2, 8)), levels = 8)


# We'll do the below steps for both `roc_auc` and `accuracy` in order to use both in our project report

#######################################################################################################


# ROC AUC

# Next, let's fit the models to our folded data using `tune_grid()`
sunset_rf_tune_res_auc <- tune_grid(
  sunset_rand_forest_wf, 
  resamples = sunset_folds, 
  grid = sunset_rf_param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)

# find the `roc_auc` of best-performing random forest tree on the folds
# use `collect_metrics()` and `arrange()`
sunset_best_rd_auc <- dplyr::arrange(collect_metrics(sunset_rf_tune_res_auc), desc(mean))
head(sunset_best_rd_auc)

# select the random forest with the best `roc_auc`
best_rf_complexity_auc <- select_best(sunset_rf_tune_res_auc)

# use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_rf_final_auc <- finalize_workflow(sunset_rand_forest_wf, best_rf_complexity_auc)
sunset_rf_final_fit_auc <- fit(sunset_rf_final_auc, data = sunset_train)


#######################################################################################################


# ACCURACY

# Next, let's fit the models to our folded data using `tune_grid()`
sunset_rf_tune_res_accuracy <- tune_grid(
  sunset_rand_forest_wf, 
  resamples = sunset_folds, 
  grid = sunset_rf_param_grid, 
  metrics = metric_set(accuracy)
)

# find the `accuracy` of best-performing random forest tree on the folds
# use `collect_metrics()` and `arrange()`
sunset_best_rd_accuracy <- dplyr::arrange(collect_metrics(sunset_rf_tune_res_accuracy), desc(mean))
head(sunset_best_rd_accuracy)

# select the random forest with the best `accuracy`
best_rf_complexity_accuracy <- select_best(sunset_rf_tune_res_accuracy)

#use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_rf_final_accuracy <- finalize_workflow(sunset_rand_forest_wf, best_rf_complexity_accuracy)
sunset_rf_final_fit_accuracy <- fit(sunset_rf_final_accuracy, data = sunset_train)



# saving data to load into rmd file
save(sunset_rf_tune_res_auc, sunset_rf_final_fit_auc, sunset_best_rd_auc,
     sunset_rf_tune_res_accuracy, sunset_rf_final_fit_accuracy, sunset_best_rd_accuracy, 
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Random_Forest.rda")
