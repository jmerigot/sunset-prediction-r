##############################
### Lasso Regression Model ###
##############################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")


sunset_spec <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

sunset_workflow <- workflow() %>% 
  add_recipe(sunset_recipe) %>% 
  add_model(sunset_spec)

pen_mix_grid <- grid_regular(penalty(range = c(-5, 5)), mixture(range = c(0,1)), levels = 10)
pen_mix_grid


#Let's fit the models to our folded data using `tune_grid()`.

lasso_tune_res <- tune_grid(
  sunset_workflow,
  resamples = sunset_folds, 
  grid = pen_mix_grid
)

autoplot(lasso_tune_res)


#Let's use `select_best()` to choose the model that has the optimal `roc_auc`.

collect_metrics(lasso_tune_res)

best_penalty <- select_best(lasso_tune_res, metric = "roc_auc")
best_penalty


#Then we'll use `finalize_workflow()`, `fit()`, and `augment()` to fit the model to the training set and evaluate its performance on the testing set.

lasso_final <- finalize_workflow(sunset_workflow, best_penalty)

lasso_final_fit <- fit(lasso_final, data = sunset_train)

augment(lasso_final_fit, new_data = sunset_test) %>%
  accuracy(truth = good_sunset, estimate = .pred_class)


#Almost there! Now let's calculate the overall ROC AUC on the testing set.

roc_lasso <- augment(lasso_final_fit, sunset_test)

roc_lasso %>%
  roc_auc(good_sunset, .pred_No)



save(lasso_tune_res, roc_lasso, lasso_final_fit, 
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Lasso_Regression.rda")