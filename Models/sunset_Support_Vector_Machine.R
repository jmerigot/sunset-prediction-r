##############################
### Support Vector Machine ###
##############################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")

set.seed(8488)


# Setup a support vector machine model and workflow. Use the `kernlab` engine and also tune `cost`
sunset_svm_rbf_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("kernlab")

sunset_svm_rbf_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(sunset_svm_rbf_spec %>% set_args(cost = tune()))


# creating parameter grid to tune ranges of hyper parameters
sunset_svm_param_grid <- grid_regular(cost(range = c(-10, 5)), levels = 10)



# We'll do the below steps for both `roc_auc` and `accuracy` in order to use both in our project report

#######################################################################################################


# ROC AUC

# Next, let's fit the models to our folded data using `tune_grid()`
sunset_svm_auc_tune_res <- tune_grid(
  sunset_svm_rbf_wf, 
  resamples = sunset_folds, 
  grid = sunset_svm_param_grid,
  metrics = metric_set(yardstick::roc_auc)
)

# find the `roc_auc` of best-performing support vector machine on the folds
# use `collect_metrics()` and `arrange()`
sunset_best_svm_auc <- dplyr::arrange(collect_metrics(sunset_svm_auc_tune_res), desc(mean))
head(sunset_best_svm_auc)

# select the support vector machine with the best `roc_auc`
sunset_best_svm_complexity_auc <- select_best(sunset_svm_auc_tune_res)

# use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_svm_final_auc <- finalize_workflow(sunset_svm_rbf_wf, sunset_best_svm_complexity_auc)
sunset_svm_final_fit_auc <- fit(sunset_svm_final_auc, data = sunset_train)


#######################################################################################################


# ACCURACY

# Next, let's fit the models to our folded data using `tune_grid()`
sunset_svm_accuracy_tune_res <- tune_grid(
  sunset_svm_rbf_wf, 
  resamples = sunset_folds, 
  grid = sunset_svm_param_grid,
  metrics = metric_set(accuracy)
)

# find the `accuracy` of best-performing support vector machine on the folds
# use `collect_metrics()` and `arrange()`
sunset_best_svm_accuracy <- dplyr::arrange(collect_metrics(sunset_svm_accuracy_tune_res), desc(mean))
head(sunset_best_svm_accuracy)

# select the support vector machine with the best `accuracy`
best_svm_complexity_accuracy <- select_best(sunset_svm_accuracy_tune_res)

#use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_svm_final_accuracy <- finalize_workflow(sunset_svm_rbf_wf, best_svm_complexity_accuracy)
sunset_svm_final_fit_accuracy <- fit(sunset_svm_final_accuracy, data = sunset_train)



# Saving data to load into main rmd file
save(sunset_svm_auc_tune_res, sunset_best_svm_auc, sunset_svm_final_fit_auc, 
     sunset_best_svm_accuracy, sunset_svm_final_fit_accuracy,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Support_Vector_Machine.rda")