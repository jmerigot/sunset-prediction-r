####################################
### Linear Discriminant Analysis ###
####################################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")


lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

lda_wkflow <- workflow() %>% 
  add_model(lda_mod) %>% 
  add_recipe(sunset_recipe)

sunset_lda_fit <- fit(lda_wkflow, sunset_train)
predict(sunset_lda_fit, new_data = sunset_train, type="prob")# %>% View()


lda_kfold_fit <- fit_resamples(lda_wkflow, sunset_folds)

collect_metrics(lda_kfold_fit)

roc_lda <- augment(sunset_lda_fit, sunset_test)


# Plotting an ROC curve on the testing data and calculating the area under the curve (AUC).

# plotting the ROC curve
roc_lda %>%
  roc_curve(good_sunset, .pred_No) %>%
  autoplot()

# calculating the AUC of the curve
roc_lda %>%
  roc_auc(good_sunset, .pred_No)


save(sunset_lda_fit, roc_lda,  
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Linear_Discriminant.rda")