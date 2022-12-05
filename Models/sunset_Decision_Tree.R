###########################
### Decision Tree Model ###
###########################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")

set.seed(8488)


# Setup a decision tree model and workflow, tune the `cost_complexity` hyper parameter
sunset_dec_tree_spec <- decision_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart")

sunset_dec_tree_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(sunset_dec_tree_spec %>% set_args(cost_complexity = tune())) 


# creating parameter grid to tune ranges of hyper parameters
sunset_dt_param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)


# fit the models to our folded data using `tune_grid()`
sunset_dt_tune_res <- tune_grid(
  sunset_dec_tree_wf, 
  resamples = sunset_folds, 
  grid = sunset_dt_param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)


# find the `roc_auc` of best-performing pruned decision tree on the folds
# use `collect_metrics()` and `arrange()`
sunset_best_pruned_tree <- dplyr::arrange(collect_metrics(sunset_dt_tune_res), desc(mean))
sunset_best_pruned_tree


# select the decision tree with the best `roc_auc`
sunset_dt_best_complexity <- select_best(sunset_dt_tune_res)


#use `finalize_workflow()` and `fit()` to fit the model to the training set
sunset_dt_final <- finalize_workflow(sunset_dec_tree_wf, sunset_dt_best_complexity)
sunset_dt_final_fit <- fit(sunset_dt_final, data = sunset_train)



# saving data to load into rmd file
save(sunset_dt_tune_res, sunset_dt_final_fit,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Decision_Tree.rda")