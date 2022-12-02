###################################
### Logistical Regression Model ###
###################################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_model_setup.rda")


log_reg <- logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")

log_wkflow <- workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(sunset_recipe)

sunset_log_fit <- fit(log_wkflow, sunset_train)
predict(log_fit, new_data = sunset_train, type="prob")# %>% View()

sunset_log_fit %>% 
  tidy()


log_kfold_fit <- fit_resamples(log_wkflow, sunset_folds)

collect_metrics(log_kfold_fit)

roc_log <- augment(sunset_log_fit, sunset_test)


# Plotting an ROC curve on the testing data and calculating the area under the curve (AUC).

# plotting the ROC curve
roc_log %>%
  roc_curve(good_sunset, .pred_No) %>%
  autoplot()

# calculating the AUC of the curve
roc_log %>%
  roc_auc(good_sunset, .pred_No)


save(sunset_log_fit, roc_log, log_kfold_fit,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Logistic_Regression.rda")