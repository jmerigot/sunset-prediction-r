####################################
### Linear Discriminant Analysis ###
####################################

load("C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Model_Setup.rda")

set.seed(8488)


# Setup linear discriminant model and workflow
sunset_lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

sunset_lda_wkflow <- workflow() %>% 
  add_model(sunset_lda_mod) %>% 
  add_recipe(sunset_recipe)


# fitting model to the training data
sunset_lda_fit <- fit(sunset_lda_wkflow, sunset_train)
predict(sunset_lda_fit, new_data = sunset_train, type="prob")


# fitting model to the folds
sunset_lda_kfold_fit <- fit_resamples(sunset_lda_wkflow, sunset_folds)
collect_metrics(sunset_lda_kfold_fit)


# saving data to load into rmd file
save(sunset_lda_fit, sunset_lda_kfold_fit, 
     file = "C:/Users/jules/OneDrive/Desktop/Sunset-Prediction-Project/RDA/sunset_Linear_Discriminant.rda")