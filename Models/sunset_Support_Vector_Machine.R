##############################
### Support Vector Machine ###
##############################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")



svm_rbf_spec <- svm_rbf() %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_rbf_wf <- workflow() %>%
  add_recipe(sunset_recipe) %>%
  add_model(svm_rbf_spec %>% set_args(cost = tune()))


param_grid <- grid_regular(cost(range = c(-10, 5)), levels = 10)

svm_tune_res <- tune_grid(
  svm_rbf_wf, 
  resamples = sunset_folds, 
  grid = param_grid,
  metrics = metric_set(yardstick::roc_auc)
)

autoplot(svm_tune_res)


best_svm <- dplyr::arrange(collect_metrics(svm_tune_res), desc(mean))
head(best_svm)


best_svm_complexity <- select_best(svm_tune_res)

svm_final <- finalize_workflow(svm_rbf_wf, best_svm_complexity)

svm_final_fit <- fit(svm_final, data = sunset_train)



save(svm_tune_res, best_svm, svm_final_fit,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Support_Vector_Machine.rda")