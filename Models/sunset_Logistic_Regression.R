###################################
### Logistical Regression Model ###
###################################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_model_setup.rda")

set.seed(8488)


# Setup logistical model and workflow
sunset_log_reg <- logistic_reg() %>% 
  set_engine("glm") %>%
  set_mode("classification")

sunset_log_wkflow <- workflow() %>% 
  add_model(sunset_log_reg) %>% 
  add_recipe(sunset_recipe)


# fitting model to the training data
sunset_log_fit <- fit(sunset_log_wkflow, sunset_train)
predict(sunset_log_fit, new_data = sunset_train, type="prob")


# fitting model to the folds
sunset_log_kfold_fit <- fit_resamples(sunset_log_wkflow, sunset_folds)
collect_metrics(sunset_log_kfold_fit)


# saving data to load into rmd file
save(sunset_log_fit, sunset_log_kfold_fit,
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Logistic_Regression.rda")