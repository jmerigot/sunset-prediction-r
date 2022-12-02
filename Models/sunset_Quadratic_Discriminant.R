#######################################
### Quadratic Discriminant Analysis ###
#######################################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")


qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

qda_wkflow <- workflow() %>% 
  add_model(qda_mod) %>% 
  add_recipe(sunset_recipe)

sunset_qda_fit <- fit(qda_wkflow, sunset_train)
predict(sunset_qda_fit, new_data = sunset_train, type="prob")# %>% View()


qda_kfold_fit <- fit_resamples(qda_wkflow, sunset_folds)

collect_metrics(qda_kfold_fit)

roc_qda <- augment(sunset_qda_fit, sunset_test)


# Plotting an ROC curve on the testing data and calculating the area under the curve (AUC).

# plotting the ROC curve
roc_qda %>%
  roc_curve(good_sunset, .pred_No) %>%
  autoplot()

# calculating the AUC of the curve
roc_qda %>%
  roc_auc(good_sunset, .pred_No)


save(sunset_qda_fit, roc_qda,  
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Quadratic_Discriminant.rda")