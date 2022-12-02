###########################
### Random Forest Model ###
###########################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")



#Now we'll set up a random forest model and workflow. Use the `ranger` engine and set 
#`importance = "impurity"`. Let's also tune `mtry`, `trees`, and `min_n`.

rf_spec <- rand_forest(mtry = tune(), trees = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rand_tree_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(rf_spec)


#We'll then create a regular grid with 8 levels each. We'll choose plausible ranges for each hyperparameter.

forest_param_grid <- grid_regular(mtry(range = c(2, 15)), trees(range = c(2, 10)), 
                                  min_n(range = c(2, 10)), levels = 8)


#Next, let's specify `roc_auc` as a metric, then tune the model and print an `autoplot()` of the results.

forest_tune_res <- tune_grid(
  rand_tree_wf, 
  resamples = sunset_folds, 
  grid = forest_param_grid, 
  metrics = metric_set(yardstick::roc_auc)
)

autoplot(forest_tune_res)


#Let's once again find the `roc_auc` of our best-performing random forest tree model on the folds. 
#We'll use `collect_metrics()` and `arrange()`.

best_rd_tree <- dplyr::arrange(collect_metrics(forest_tune_res), desc(mean))
head(best_rd_tree)



best_forest_complexity <- select_best(forest_tune_res)

rand_tree_final <- finalize_workflow(rand_tree_wf, best_forest_complexity)

rand_tree_final_fit <- fit(rand_tree_final, data = sunset_train)



save(forest_tune_res, rand_tree_final_fit, best_rd_tree,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Random_Forest.rda")
