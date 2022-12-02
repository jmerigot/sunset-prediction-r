###########################
### Decision Tree Model ###
###########################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")


#First, we'll set up a decision tree model and workflow. We'll tune the `cost_complexity` hyper parameter.

tree_spec <- decision_tree() %>%
  set_engine("rpart")

class_tree_spec <- tree_spec %>%
  set_mode("classification")

class_tree_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) 


#Then, we'll use the levels `range = c(-3, -1)`. We'll also specify that the metric we want to optimize is `roc_auc`.

param_grid <- grid_regular(cost_complexity(range = c(-3, -1)), levels = 10)

dt_tune_res <- tune_grid(
  class_tree_wf, 
  resamples = sunset_folds, 
  grid = param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)

autoplot(dt_tune_res)


#Let's find the `roc_auc` of our best-performing pruned decision tree on the folds. We'll use `collect_metrics()` and `arrange()`.

best_pruned_tree <- dplyr::arrange(collect_metrics(dt_tune_res), desc(mean))
best_pruned_tree


#Using `rpart.plot`, we'll fit and visualize our best-performing pruned decision tree with the training set.

best_complexity <- select_best(dt_tune_res)

class_tree_final <- finalize_workflow(class_tree_wf, best_complexity)

class_tree_final_fit <- fit(class_tree_final, data = sunset_train)



save(dt_tune_res, class_tree_final_fit,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Decision_Tree.rda")