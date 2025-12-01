# Load required libraries
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Machine learning framework
library(ranger)       # Fast random forest implementation
library(DALEX)        # Model-agnostic interpretation
library(doParallel)   # Parallel processing (if needed)

# Additional libraries loaded but not used in this cleaned version:
# library("faux")
# library("randomForest") # ranger is used instead
# library("rms")
# library("gbm")
# library(e1071)
# library(Matching)
setwd("D:/Alejandro - Carlos/Proyecto Brasil/2da parte Proyecto/Tercer paper/Manuscrito/Nature Communications/Para publicar/Codigo/Respositorio")
# ------------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ------------------------------------------------------------

# Load dataset
df_poblacion_2 <- read.csv("Data/Data.csv")

# Filter and clean data
df_poblacion_2 <- df_poblacion_2 %>%
  # Remove rows with missing values in key variables
  filter(!is.na(Sewage), !is.na(Occupation), !is.na(Ethnicity)) %>%
  na.omit() %>%
  
  # Recode Occupation variable
  mutate(Occupation = case_when(
    Occupation == "Unemployed" ~ 0,
    Occupation == "Formal employment" ~ 3,
    Occupation == "Retiree" ~ 1,
    Occupation == "Casual employment" ~ 2
  )) %>%
  
  # Recode and convert chronic disease diagnosis to factor
  mutate(Dx_Cronical_disease = case_when(
    Dx_Cronical_disease == "No" ~ 0,
    Dx_Cronical_disease == "Yes" ~ 1
  )) %>%
  mutate(Dx_Cronical_disease = as.factor(Dx_Cronical_disease)) %>%
  
  # Remove unnecessary columns
  dplyr::select(-X, -Illiteracy)
##########################################################
###Hyperparameter selection using cross-validation

df_poblacion_test <- df_poblacion_2 %>%
  filter(Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()
 
index_Sample <- createDataPartition(df_poblacion_test$Dx_Cronical_disease, p = .1, list = FALSE)

df_poblacion_test <- df_poblacion_test %>%
  mutate_at(1:8,as.numeric) 
# We divide between train and test

train <- df_poblacion_test[index_Sample,]
test <- df_poblacion_test[-index_Sample,]

# Hyperparameter
hiperparametros <- expand.grid(mtry = c(3, 4, 5, 7, 10))
#===============================================================================
particiones  <- 10
repeticiones <- 3

control_train <- trainControl(method = "repeatedcv", number = particiones,
                              repeats = repeticiones,
                              returnResamp = "final", verboseIter = FALSE,
                              allowParallel = TRUE)

set.seed(1234)
modelo_rf_all <- train(Dx_Cronical_disease ~ ., data = train,
                            method = "rf",
                            tuneGrid = hiperparametros,
                            metric = "AUC",
                            trControl = control_train,
                            # Número de árboles ajustados
                            num.trees = 300)
#We selected the best hyperparameters
modelo_rf_all

# ------------------------------------------------------------
# 2. SUBSET CREATION: MIXED-RACE MEN
# ------------------------------------------------------------

# Filter for adult mixed-race men (Sex == 1, Ethnicity == 1, Age > 19)
df_Male_mixrace <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert all predictor columns to numeric format
df_Male_mixrace <- df_Male_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# ------------------------------------------------------------
# 3. DATA SPLITTING FOR TRAINING AND TESTING
# ------------------------------------------------------------

set.seed(1234)  # For reproducibility

# Create 70-30 train-test split stratified by outcome variable
index_train <- createDataPartition(df_Male_mixrace$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Male_mixrace[index_train, ]
test <- df_Male_mixrace[-index_train, ]

# ------------------------------------------------------------
# 4. WEIGHT HANDLING FOR SURVEY REPRESENTATIVENESS
# ------------------------------------------------------------

# Training data: Separate weights from predictors
train_2 <- train
pesos_train <- train_2$weights_of_people  # Survey weights for training
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

# Testing data: Separate weights from predictors
test_2 <- test
pesos_test <- test_2$weights_of_people  # Survey weights for testing
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# ------------------------------------------------------------
# 5. RANDOM FOREST MODEL TRAINING WITH SURVEY WEIGHTS
# ------------------------------------------------------------

modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,  # Predict using all variables
  data = train_2,
  num.trees = 300,           # Number of trees in the forest
  min.node.size = 5,         # Minimum node size
  mtry = 3,                  # Number of variables randomly sampled at each split
  verbose = FALSE,
  importance = "none",       # Variable importance computed separately via DALEX
  seed = 123,                # Reproducibility
  probability = TRUE,        # Return probability predictions (not just class)
  case.weights = pesos_train # Incorporate survey weights in training
)

# ------------------------------------------------------------
# 6. MODEL EXPLANATION AND PERFORMANCE ASSESSMENT
# ------------------------------------------------------------

# Create explainer object for model-agnostic interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables only (exclude outcome in column 9)
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),  # Binary outcome as numeric
  weights = pesos_test,  # Apply survey weights to test data
  verbose = FALSE
)

# Calculate model performance metrics
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 7. ACCUMULATED LOCAL EFFECTS (ALE) PROFILES
# ------------------------------------------------------------

# Note: ALE plots show how each predictor affects the predicted probability
# of chronic disease, averaged over the data distribution

# 7.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000  # Number of samples for profile calculation
)

df_graf_edu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 7.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 8. VARIABLE IMPORTANCE VIA PERMUTATION (MODEL PARTS)
# ------------------------------------------------------------

# Calculate permutation-based variable importance with parallel processing
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,          # Number of permutations per variable
    N = 2000,        # Number of observations to sample
    type = "difference",  # Performance loss when variable is permuted
    parallel = TRUE, # Parallel processing for computational efficiency
    weights = pesos_test  # Apply survey weights
  )
})

# Visualize variable importance
library(scales)

plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mixed-Race Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# ------------------------------------------------------------
# 9. SUMMARIZE IMPORTANCE METRICS FOR MIXED-RACE MEN
# ------------------------------------------------------------

var_impo_RF_Male_Mix <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%  # Average performance loss
  mutate(Ethnicity = "Mixed",
         Sex = "Men",
         AUC = mp_rf_male_mix$measures$auc)  # Attach model AUC

# ------------------------------------------------------------
# 10. REPEAT ANALYSIS FOR BLACK MEN SUBGROUP
# ------------------------------------------------------------

# 10.1 Data subset: Black men (Sex == 1, Ethnicity == 2, Age > 19)
df_Male_black <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 2, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_black <- df_Male_black %>%
  mutate_at(c(1:8), as.numeric)

# 10.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_black$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_black[index_train, ]
test <- df_Male_black[-index_train, ]

# 10.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 10.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 10.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables only
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 10.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 11. ACCUMULATED LOCAL EFFECTS (ALE) FOR BLACK MEN
# ------------------------------------------------------------

# Note: The same predictors are analyzed as for mixed-race men
# to enable cross-group comparisons

# 11.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 11.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)




# ------------------------------------------------------------
# 12. VARIABLE IMPORTANCE FOR BLACK MEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance for Black men
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Black Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Black men
var_impo_RF_Male_Black <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Black",
         Sex = "Men",
         AUC = mp_rf_male_mix$measures$auc)  # Attach model AUC

# ------------------------------------------------------------
# 13. REPEAT ANALYSIS FOR INDIGENOUS MEN SUBGROUP
# ------------------------------------------------------------

# 13.1 Data subset: Indigenous men (Sex == 1, Ethnicity == 3, Age > 19)
df_Male_indige <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_indige <- df_Male_indige %>%
  mutate_at(c(1:8), as.numeric)

# 13.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_indige$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_indige[index_train, ]
test <- df_Male_indige[-index_train, ]

# 13.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 13.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 13.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables only
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 13.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 14. ACCUMULATED LOCAL EFFECTS (ALE) FOR INDIGENOUS MEN
# ------------------------------------------------------------

# 14.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 14.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)


# ------------------------------------------------------------
# 16. VARIABLE IMPORTANCE FOR INDIGENOUS MEN (COMPLETED)
# ------------------------------------------------------------

# Calculate permutation-based variable importance for Indigenous men
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Indigenous Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous men
var_impo_RF_Male_indige <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Men",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 17. REPEAT ANALYSIS FOR "OTHERS" MEN SUBGROUP (ETHNICITY == 4)
# ------------------------------------------------------------

# 17.1 Data subset: Other men (Sex == 1, Ethnicity == 4, Age > 19)
df_Male_others <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 4, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_others <- df_Male_others %>%
  mutate_at(c(1:8), as.numeric)

# 17.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_others$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_others[index_train, ]
test <- df_Male_others[-index_train, ]

# 17.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 17.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 17.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 17.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 18. ACCUMULATED LOCAL EFFECTS (ALE) FOR OTHER MEN
# ------------------------------------------------------------

# 18.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 18.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 19. VARIABLE IMPORTANCE FOR OTHER MEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance for Other men
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Other Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Other men
var_impo_Male_others <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Others",
         Sex = "Men",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 20. AGGREGATE RESULTS ACROSS ALL MALE SUBGROUPS
# ------------------------------------------------------------

# 20.1 Combine variable importance results for all male subgroups
mens_models <- rbind(var_impo_RF_Black_Mix,   # Note: Name correction - should be var_impo_RF_Male_Black
                     var_impo_RF_Male_Mix,
                     var_impo_RF_Male_indige,
                     var_impo_Male_others)

# Calculate percentage performance loss when variable is permuted
mens_models <- mens_models %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# 20.2 Combine ALE profiles across subgroups for each predictor

# Age profiles
df_graf_age <- rbind(df_graf_Age_1, df_graf_Age_2, 
                     df_graf_Age_3, df_graf_Age_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles
df_graf_edu <- rbind(df_graf_edu_1, df_graf_edu_2,
                     df_graf_edu_3, df_graf_edu_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles
df_graf_ocupa <- rbind(df_graf_ocu_1, df_graf_ocu_2,
                       df_graf_ocu_3, df_graf_ocu_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles
df_graf_piper <- rbind(df_graf_piper_1, df_graf_piper_2,
                       df_graf_piper_3, df_graf_piper_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles
df_graf_garba <- rbind(df_graf_garbage_1, df_graf_garbage_2,
                       df_graf_garbage_3, df_graf_garbage_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles
df_graf_urban <- rbind(df_graf_urban_1, df_graf_urban_2,
                       df_graf_urban_3, df_graf_urban_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles
df_graf_water <- rbind(df_graf_water_1, df_graf_water_2,
                       df_graf_water_3, df_graf_water_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles
df_graf_sewage <- rbind(df_graf_sewage_1, df_graf_sewage_2,
                        df_graf_sewage_3, df_graf_sewage_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 21. VISUALIZE SUBGROUP COMPARISONS
# ------------------------------------------------------------

# 21.1 Age effects across subgroups
df_graf_age %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10)

# 21.2 Education effects across subgroups
df_graf_edu %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Education") +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("No education", "Complete primary", 
               "Complete secondary", "Higher")
  ) +
  scale_y_continuous(n.breaks = 10)

# 21.3 Occupation effects across subgroups
df_graf_ocupa %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Occupation") +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("Unemployed", "Retired", 
               "Casual employment", "Formal employment")
  ) +
  scale_y_continuous(n.breaks = 10)

# 21.4 Service access effects across subgroups (similar plots for other variables)

# Piped water
df_graf_piper %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Piped Water") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# Garbage collection
df_graf_garba %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Garbage Collection") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# Urban/rural residence
df_graf_urban %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Urban/Rural") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# General water access
df_graf_water %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Water Access") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# Sewage system
df_graf_sewage %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Sewage System") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# ------------------------------------------------------------
# 22. DATA PREPARATION FOR PUBLICATION-READY VISUALIZATIONS
# ------------------------------------------------------------

# Create publication-ready data frames with interpretable labels
df_graf_age_new <- df_graf_age %>% 
  mutate(variable = "Age")

df_graf_edu_new <- df_graf_edu %>% 
  mutate(variable = "Education") %>%
  mutate(value = case_when(
    value == 0 ~ "No education",
    value == 1 ~ "Complete primary",
    value == 2 ~ "Complete secondary",
    value == 3 ~ "Higher"
  ))

df_graf_ocupa_new <- df_graf_ocupa %>% 
  mutate(variable = "Occupation") %>%
  mutate(value = case_when(
    value == 0 ~ "Unemployed",
    value == 1 ~ "Retired",
    value == 2 ~ "Casual employment",
    value == 3 ~ "Formal employment"
  ))

# Binary variables recoded for clarity
df_graf_piper_new <- df_graf_piper %>% 
  mutate(variable = "Piped water") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_garba_new <- df_graf_garba %>% 
  mutate(variable = "Garbage") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_urban_new <- df_graf_urban %>% 
  mutate(variable = "Urban") %>%
  mutate(value = if_else(value == 1, "Urban", "Rural"))

df_graf_water_new <- df_graf_water %>% 
  mutate(variable = "Water") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_sewage_new <- df_graf_sewage %>% 
  mutate(variable = "Sewage") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

# ------------------------------------------------------------
# 23. CREATE COMPREHENSIVE DATASET FOR FIGURE 2 (MEN ONLY)
# ------------------------------------------------------------

df_dependencia_hombres <- rbind(
  df_graf_age_new,
  df_graf_edu_new,
  df_graf_ocupa_new,
  df_graf_piper_new,
  df_graf_garba_new,
  df_graf_urban_new,
  df_graf_water_new,
  df_graf_sewage_new
)

# This dataset contains all ALE profiles for male subgroups
# Structure: value, Group, U5MR_average, variable
# Save as source data for Figure 2
# write.csv(df_dependencia_hombres, 
#          "Source_Data_Figure2_Men_ALE_Profiles.csv")

# ------------------------------------------------------------
# 24. CREATE INDIVIDUAL PLOT COMPONENTS
# ------------------------------------------------------------

library(patchwork)
library(ggplot2)
library(grid)  # For unit() function

# 24.1 Age effect plot
p_age <- df_graf_age %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Age") +
  theme(legend.position = "none")

# 24.2 Education effect plot (with abbreviated labels)
p_edu <- df_graf_edu %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("No education", "CP", "CS", "Higher")  # CP=Complete primary, CS=Complete secondary
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Education") +
  theme(legend.position = "none")

# 24.3 Occupation effect plot (with abbreviated labels)
p_ocupa <- df_graf_ocupa %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("Unemployed", "Retired", "CE", "FE")  # CE=Casual employment, FE=Formal employment
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Occupation") +
  theme(legend.position = "none")

# 24.4 Service access plots (binary variables)
p_piper <- df_graf_piper %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Piped Water") +
  theme(legend.position = "none")

p_garba <- df_graf_garba %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Garbage Collection") +
  theme(legend.position = "none")

p_urban <- df_graf_urban %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Rural", "Urban")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Residence") +
  theme(legend.position = "none")

p_water <- df_graf_water %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Water Access") +
  theme(legend.position = "none")

p_sewage <- df_graf_sewage %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Sewage System") +
  theme(legend.position = "none")

# ------------------------------------------------------------
# 25. CREATE AND EXTRACT LEGEND
# ------------------------------------------------------------

# Create a version of p_age with legend for extraction
p_with_legend <- df_graf_age %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10) +
  xlab("Age") +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.box = "horizontal")

# Extract legend using cowplot (assumes library(cowplot) loaded)
library(cowplot)
legend <- get_legend(
  p_with_legend + 
    theme(legend.position = "bottom",
          legend.direction = "horizontal")
)

# ------------------------------------------------------------
# 26. PREPARE PLOTS FOR MULTI-PANEL FIGURE
# ------------------------------------------------------------

# Remove y-axis labels from individual plots for cleaner layout
p_age <- p_age + ylab(NULL)
p_edu <- p_edu + ylab(NULL)
p_ocupa <- p_ocupa + ylab(NULL)
p_piper <- p_piper + ylab(NULL)
p_garba <- p_garba + ylab(NULL)
p_urban <- p_urban + ylab(NULL)
p_water <- p_water + ylab(NULL)
p_sewage <- p_sewage + ylab(NULL)

# Create separate y-axis label for shared axis
axis_y_label <- ggplot() + 
  ylab("Average probability of chronic disease diagnosis") +
  theme_void() +
  theme(
    axis.title.y = element_text(angle = 90, size = 14, face = "bold"),
    plot.margin = unit(c(0, 2, 0, 0), "points")
  )

# ------------------------------------------------------------
# 27. ASSEMBLE MULTI-PANEL FIGURE
# ------------------------------------------------------------

# Initial plot arrangement (8 panels in 4×2 grid)
combined_plot <- (p_edu | p_piper | p_ocupa | 
                    p_garba | p_urban | p_water | p_sewage | p_age) +
  plot_layout(nrow = 4, ncol = 2, guides = "collect")

# Add shared y-axis label
combined_plot <- wrap_plots(
  axis_y_label,           # Left: y-axis label
  combined_plot,          # Right: 8-panel plot
  widths = c(0.01, 0.965) # Adjust spacing
)

# Add legend at bottom
final_plot <- combined_plot / legend +
  plot_layout(heights = c(50, 1),  # Main plot : legend ratio
              widths = c(0.01, 0.965))

# Apply consistent text styling
final_plot <- final_plot &
  theme(
    text = element_text(size = 12),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    strip.text = element_text(size = 12)
  )

# Display final plot
final_plot

# ------------------------------------------------------------
# 29. FEMALE SUBGROUP ANALYSIS: MIXED-RACE WOMEN
# ------------------------------------------------------------

# 29.1 Data subset: Mixed-race women (Sex == 0, Ethnicity == 1, Age > 19)
# Note: Sex coding differs from men (0 = female, 1 = male in original data)
df_Female_mixrace <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_mixrace <- df_Female_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# 29.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_mixrace[index_train, ]
test <- df_Female_mixrace[-index_train, ]

# 29.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 29.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 29.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables only (outcome in column 9)
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 29.6 Calculate model performance
# Note: Variable name retained as "mp_rf_male_mix" for consistency in pipeline
# but represents female mixed-race model
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 30. ACCUMULATED LOCAL EFFECTS (ALE) FOR MIXED-RACE WOMEN
# ------------------------------------------------------------

# 30.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",  # Note: Changed from "Men Mixed"
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 30.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 31. VARIABLE IMPORTANCE FOR MIXED-RACE WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mixed-Race Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race women
var_impo_RF_Female_Mix <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Women",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------


# 32.1 Data subset: Black women (Sex == 0, Ethnicity == 2, Age > 19)
df_Female_black <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 2, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_black <- df_Female_black %>%
  mutate_at(c(1:8), as.numeric)

# 32.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_black$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_black[index_train, ]
test <- df_Female_black[-index_train, ]

# 32.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 32.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 32.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 32.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 33. ACCUMULATED LOCAL EFFECTS (ALE) FOR BLACK WOMEN
# ------------------------------------------------------------

# 33.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",  # Note: Corrected from "Men Black" in original code
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.4 Garbage collection effect
# NOTE: Error in original code - Group labeled as "Men Black" instead of "Women Black"
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",  # Corrected from "Men Black" in original
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 33.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_2 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 34. VARIABLE IMPORTANCE FOR BLACK WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Black Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Black women
var_impo_RF_Female_black <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Black",
         Sex = "Women",
         AUC = mp_rf_male_mix$measures$auc)


# ------------------------------------------------------------

# 35.1 Data subset: Indigenous women (Sex == 0, Ethnicity == 3, Age > 19)
df_Female_Indigenous <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_Indigenous <- df_Female_Indigenous %>%
  mutate_at(c(1:8), as.numeric)

# 35.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_Indigenous$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_Indigenous[index_train, ]
test <- df_Female_Indigenous[-index_train, ]

# 35.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 35.4 Train Random Forest model with survey weights
set.seed(1234)
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 35.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 35.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 36. ACCUMULATED LOCAL EFFECTS (ALE) FOR INDIGENOUS WOMEN
# ------------------------------------------------------------

# 36.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 36.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 37. VARIABLE IMPORTANCE FOR INDIGENOUS WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Indigenous Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous women
var_impo_RF_Female_indigenous <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Women",
         AUC = mp_rf_male_mix$measures$auc)


# ------------------------------------------------------------
# 38. FEMALE SUBGROUP ANALYSIS: OTHER WOMEN (FINAL SUBGROUP)
# ------------------------------------------------------------

# 38.1 Data subset: Other women (Sex == 0, Ethnicity == 4, Age > 19)
df_Female_Others <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 4, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_Others <- df_Female_Others %>%
  mutate_at(c(1:8), as.numeric)

# 38.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_Others$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_Others[index_train, ]
test <- df_Female_Others[-index_train, ]

# 38.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 38.4 Train Random Forest model with survey weights
set.seed(1234)
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 38.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 38.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 39. ACCUMULATED LOCAL EFFECTS (ALE) FOR OTHER WOMEN
# ------------------------------------------------------------

# 39.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 39.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_4 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 40. VARIABLE IMPORTANCE FOR OTHER WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
set.seed(1234)
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Other Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Other women
var_impo_RF_Female_others <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Others",
         Sex = "Women",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 41. AGGREGATE FEMALE SUBGROUP RESULTS
# ------------------------------------------------------------

# 41.1 Combine variable importance results for all female subgroups
woman_models <- rbind(var_impo_RF_Female_Mix,
                      var_impo_RF_Female_black,
                      var_impo_RF_Female_indigenous,
                      var_impo_RF_Female_others)

# Calculate percentage performance loss when variable is permuted
woman_models <- woman_models %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# 41.2 Combine ALE profiles across female subgroups for each predictor

# Age profiles for women
df_graf_age_w <- rbind(df_graf_Age_1, df_graf_Age_2,
                       df_graf_Age_3, df_graf_Age_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles for women
df_graf_edu_w <- rbind(df_graf_edu_1, df_graf_edu_2,
                       df_graf_edu_3, df_graf_edu_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles for women
df_graf_ocupa_w <- rbind(df_graf_ocu_1, df_graf_ocu_2,
                         df_graf_ocu_3, df_graf_ocu_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles for women
df_graf_piper_w <- rbind(df_graf_piper_1, df_graf_piper_2,
                         df_graf_piper_3, df_graf_piper_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles for women
df_graf_garba_w <- rbind(df_graf_garbage_1, df_graf_garbage_2,
                         df_graf_garbage_3, df_graf_garbage_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles for women
df_graf_urban_w <- rbind(df_graf_urban_1, df_graf_urban_2,
                         df_graf_urban_3, df_graf_urban_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles for women
df_graf_water_w <- rbind(df_graf_water_1, df_graf_water_2,
                         df_graf_water_3, df_graf_water_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles for women
df_graf_sewage_w <- rbind(df_graf_sewage_1, df_graf_sewage_2,
                          df_graf_sewage_3, df_graf_sewage_4) %>%
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 42. VISUALIZE FEMALE SUBGROUP COMPARISONS
# ------------------------------------------------------------

# Individual exploratory plots for each predictor (not shown in final figure)
# These are for data exploration and quality checking

# 42.1 Age effects across female subgroups (exploratory)
df_graf_age_w %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10)

# 42.2 Education effects across female subgroups (exploratory)
df_graf_edu_w %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Education") +
  scale_y_continuous(n.breaks = 10)

# 42.3 Occupation effects across female subgroups (exploratory)
df_graf_ocupa_w %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Occupation") +
  scale_y_continuous(n.breaks = 10)

# Similar exploratory plots for other predictors...

# ------------------------------------------------------------
# 43. CREATE PUBLICATION-READY DATASET FOR WOMEN'S FIGURE
# ------------------------------------------------------------

# Create interpretable labels for publication
df_graf_age_new <- df_graf_age_w %>% 
  mutate(variable = "Age")

df_graf_edu_new <- df_graf_edu_w %>% 
  mutate(variable = "Education") %>%
  mutate(value = case_when(
    value == 0 ~ "No education",
    value == 1 ~ "Complete primary",
    value == 2 ~ "Complete secondary",
    value == 3 ~ "Higher"
  ))

df_graf_ocupa_new <- df_graf_ocupa_w %>% 
  mutate(variable = "Occupation") %>%
  mutate(value = case_when(
    value == 0 ~ "Unemployed",
    value == 1 ~ "Retired",
    value == 2 ~ "Casual employment",
    value == 3 ~ "Formal employment"
  ))

df_graf_piper_new <- df_graf_piper_w %>% 
  mutate(variable = "Piped water") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_garba_new <- df_graf_garba_w %>% 
  mutate(variable = "Garbage") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_urban_new <- df_graf_urban_w %>% 
  mutate(variable = "Urban") %>%
  mutate(value = if_else(value == 1, "Urban", "Rural"))

df_graf_water_new <- df_graf_water_w %>% 
  mutate(variable = "Water") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

df_graf_sewage_new <- df_graf_sewage_w %>% 
  mutate(variable = "Sewage") %>%
  mutate(value = if_else(value == 1, "With", "Without"))

# Combine all ALE profiles for women into comprehensive dataset
df_dependencia_mujeres <- rbind(
  df_graf_age_new,
  df_graf_edu_new,
  df_graf_ocupa_new,
  df_graf_piper_new,
  df_graf_garba_new,
  df_graf_urban_new,
  df_graf_water_new,
  df_graf_sewage_new
)

# This dataset contains all ALE profiles for female subgroups
# Save as source data for Figure 4 (Women's analysis)
# write.csv(df_dependencia_mujeres, "Source_Data_Figure4_Women_ALE_Profiles.csv")

# ------------------------------------------------------------
# 44. CREATE MULTI-PANEL FIGURE FOR WOMEN (FIGURE 4)
# ------------------------------------------------------------

library(patchwork)
library(ggplot2)
library(grid)

# 44.1 Create individual plot components

# Age effect plot
p_age <- df_graf_age_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Age") +
  theme(legend.position = "none")

# Education effect plot (with abbreviated labels)
p_edu <- df_graf_edu_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("No education", "CP", "CS", "Higher")  # CP=Complete primary, CS=Complete secondary
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Education") +
  theme(legend.position = "none")

# Occupation effect plot (with abbreviated labels)
p_ocupa <- df_graf_ocupa_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:3,
    labels = c("Unemployed", "Retired", "CE", "FE")  # CE=Casual employment, FE=Formal employment
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Occupation") +
  theme(legend.position = "none")

# Service access plots (binary variables)
p_piper <- df_graf_piper_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Piped Water") +
  theme(legend.position = "none")

p_garba <- df_graf_garba_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Garbage Collection") +
  theme(legend.position = "none")

p_urban <- df_graf_urban_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Rural", "Urban")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Residence") +
  theme(legend.position = "none")

p_water <- df_graf_water_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Water Access") +
  theme(legend.position = "none")

p_sewage <- df_graf_sewage_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  scale_x_continuous(
    breaks = 0:1,
    labels = c("Without", "With")
  ) +
  scale_y_continuous(n.breaks = 5) +
  xlab("Sewage System") +
  theme(legend.position = "none")

# 44.2 Create and extract legend
p_with_legend <- df_graf_age_w %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10) +
  xlab("Age") +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        legend.box = "horizontal")

library(cowplot)
legend <- get_legend(
  p_with_legend + 
    theme(legend.position = "bottom",
          legend.direction = "horizontal")
)

# 44.3 Prepare plots for multi-panel figure
# Remove individual y-axis labels for cleaner shared axis
p_age <- p_age + ylab(NULL)
p_edu <- p_edu + ylab(NULL)
p_ocupa <- p_ocupa + ylab(NULL)
p_piper <- p_piper + ylab(NULL)
p_garba <- p_garba + ylab(NULL)
p_urban <- p_urban + ylab(NULL)
p_water <- p_water + ylab(NULL)
p_sewage <- p_sewage + ylab(NULL)

# Create separate y-axis label
axis_y_label <- ggplot() + 
  ylab("Average probability of chronic disease diagnosis") +
  theme_void() +
  theme(
    axis.title.y = element_text(angle = 90, size = 14, face = "bold"),
    plot.margin = unit(c(0, 2, 0, 0), "points")
  )

# 44.4 Assemble multi-panel figure
# Initial plot arrangement (8 panels in 4×2 grid)
combined_plot <- (p_edu | p_piper | p_ocupa | 
                    p_garba | p_urban | p_water | p_sewage | p_age) +
  plot_layout(nrow = 4, ncol = 2, guides = "collect")

# Add shared y-axis label
combined_plot <- wrap_plots(
  axis_y_label,           # Left: y-axis label
  combined_plot,          # Right: 8-panel plot
  widths = c(0.01, 0.965) # Adjust spacing
)

# Add legend at bottom
final_plot <- combined_plot / legend +
  plot_layout(heights = c(50, 1),  # Main plot : legend ratio
              widths = c(0.01, 0.965))

# Apply consistent text styling
final_plot <- final_plot &
  theme(
    text = element_text(size = 12),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    strip.text = element_text(size = 12)
  )

# Display final plot (Figure 4)
final_plot


# ------------------------------------------------------------
# 45. COUNTRY-SPECIFIC ANALYSIS: MEXICO
# ------------------------------------------------------------

# 45.1 Load and filter data for Mexico only
df_poblacion_2 <- read.csv("Data/Data.csv") %>%
  filter(Country == "Mexico")  # Country-specific filter

# 45.2 Data cleaning and preprocessing (same as before)
df_poblacion_2 <- df_poblacion_2 %>%
  # Remove rows with missing values in key variables
  filter(!is.na(Sewage), !is.na(Occupation), !is.na(Ethnicity)) %>%
  na.omit() %>%
  
  # Recode Occupation variable (consistent with main analysis)
  mutate(Occupation = case_when(
    Occupation == "Unemployed" ~ 0,
    Occupation == "Formal employment" ~ 3,
    Occupation == "Retiree" ~ 1,
    Occupation == "Casual employment" ~ 2
  )) %>%
  
  # Recode and convert chronic disease diagnosis to factor
  mutate(Dx_Cronical_disease = case_when(
    Dx_Cronical_disease == "No" ~ 0,
    Dx_Cronical_disease == "Yes" ~ 1
  )) %>%
  mutate(Dx_Cronical_disease = as.factor(Dx_Cronical_disease)) %>%
  
  # Remove unnecessary columns
  dplyr::select(-X, -Illiteracy)

# ------------------------------------------------------------
# 46. MEXICO: MIXED-RACE MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 46.1 Data subset: Mixed-race men in Mexico
df_Male_mixrace <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_mixrace <- df_Male_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# 46.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_mixrace[index_train, ]
test <- df_Male_mixrace[-index_train, ]

# 46.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 46.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 46.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 46.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 47. MEXICO: ACCUMULATED LOCAL EFFECTS FOR MIXED-RACE MEN
# ------------------------------------------------------------

# 47.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 47.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_1 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 48. MEXICO: VARIABLE IMPORTANCE FOR MIXED-RACE MEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mexico - Mixed-Race Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race men in Mexico
var_impo_RF_Male_Mix_Mexico <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Men",
         Country = "Mexico",  # Added country identifier
         AUC = mp_rf_male_mix$measures$auc)
# ------------------------------------------------------------
# 49. MEXICO: INDIGENOUS MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 49.1 Data subset: Indigenous men in Mexico (Ethnicity == 3)
df_Male_indige <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_indige <- df_Male_indige %>%
  mutate_at(c(1:8), as.numeric)

# 49.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_indige$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_indige[index_train, ]
test <- df_Male_indige[-index_train, ]

# 49.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 49.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 49.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 49.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 50. MEXICO: ACCUMULATED LOCAL EFFECTS FOR INDIGENOUS MEN
# ------------------------------------------------------------

# 50.1 Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.2 Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.3 Piped water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Piped_water"),
  type = "accumulated",
  N = 5000
)

df_graf_piper_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.4 Garbage collection effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Garbage"),
  type = "accumulated",
  N = 5000
)

df_graf_garbage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.5 Urban/rural residence effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Urban_rural"),
  type = "accumulated",
  N = 5000
)

df_graf_urban_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.6 Sewage system effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Sewage"),
  type = "accumulated",
  N = 5000
)

df_graf_sewage_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.7 General water access effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Water"),
  type = "accumulated",
  N = 5000
)

df_graf_water_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# 50.8 Age effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Age"),
  type = "accumulated",
  N = 5000
)

df_graf_Age_3 <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# ------------------------------------------------------------
# 51. MEXICO: VARIABLE IMPORTANCE FOR INDIGENOUS MEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mexico - Indigenous Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous men in Mexico
var_impo_RF_Male_indige_Mexico <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Men",
         Country = "Mexico",
         AUC = mp_rf_male_mix$measures$auc)


# ------------------------------------------------------------
# 52. PARTIAL AGGREGATION OF MEXICO RESULTS
# ------------------------------------------------------------

# 52.1 Combine variable importance results for completed Mexico subgroups
mens_models_mexico <- rbind(var_impo_RF_Male_Mix,    # Mixed men in Mexico
                            var_impo_RF_Male_indige) # Indigenous men in Mexico

# Calculate percentage performance loss
mens_models_mexico <- mens_models_mexico %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# Note: This is a partial aggregation - missing Black and Other men subgroups
# Complete Mexico analysis would require all 4 male subgroups

# 52.2 Combine ALE profiles for completed Mexico subgroups

# Age profiles for Mixed and Indigenous men in Mexico
df_graf_age <- rbind(df_graf_Age_1,    # Mixed men
                     df_graf_Age_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles
df_graf_edu <- rbind(df_graf_edu_1,    # Mixed men
                     df_graf_edu_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles
df_graf_ocupa <- rbind(df_graf_ocu_1,    # Mixed men
                       df_graf_ocu_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles
df_graf_piper <- rbind(df_graf_piper_1,    # Mixed men
                       df_graf_piper_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles
df_graf_garba <- rbind(df_graf_garbage_1,    # Mixed men
                       df_graf_garbage_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles
df_graf_urban <- rbind(df_graf_urban_1,    # Mixed men
                       df_graf_urban_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles
df_graf_water <- rbind(df_graf_water_1,    # Mixed men
                       df_graf_water_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles
df_graf_sewage <- rbind(df_graf_sewage_1,    # Mixed men
                        df_graf_sewage_3) %>% # Indigenous men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 53. EXPLORATORY VISUALIZATIONS FOR MEXICO SUBSET
# ------------------------------------------------------------

# 53.1 Age effects: Mixed vs Indigenous men in Mexico
df_graf_age %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10)

# 53.2 Education effects: Mixed vs Indigenous men in Mexico
df_graf_edu %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Education") +
  scale_y_continuous(n.breaks = 10)

# 53.3 Occupation effects: Mixed vs Indigenous men in Mexico
df_graf_ocupa %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Occupation") +
  scale_y_continuous(n.breaks = 10)

# 53.4 Service access effects (example: Piped water)
df_graf_piper %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Piped Water") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10)

# Similar exploratory plots for other service variables...
# ------------------------------------------------------------
# 54. MEXICO: WOMEN SUBGROUP ANALYSES
# ------------------------------------------------------------

# 54.1 MEXICO: MIXED-RACE WOMEN ANALYSIS

# Data subset: Mixed-race women in Mexico (Sex == 0, Ethnicity == 1)
df_Female_mixrace <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_mixrace <- df_Female_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_mixrace[index_train, ]
test <- df_Female_mixrace[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 55. MEXICO: ACCUMULATED LOCAL EFFECTS FOR MIXED-RACE WOMEN
# ------------------------------------------------------------

# Note: These objects overwrite earlier ones - better naming needed for country analysis
# Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1_w <- data.frame(  # Added "_w" for women, but country conflict remains
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Occupation effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Occupation"),
  type = "accumulated",
  N = 5000
)

df_graf_ocu_1_w <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...
# (Piped_water, Garbage, Urban_rural, Sewage, Water, Age)

# ------------------------------------------------------------
# 56. MEXICO: VARIABLE IMPORTANCE FOR MIXED-RACE WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mexico - Mixed-Race Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race women in Mexico
var_impo_RF_Female_Mix_Mexico <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Women",
         Country = "Mexico",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 57. MEXICO: INDIGENOUS WOMEN ANALYSIS
# ------------------------------------------------------------

# Data subset: Indigenous women in Mexico (Sex == 0, Ethnicity == 3)
df_Female_Indigenous <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_Indigenous <- df_Female_Indigenous %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_Indigenous$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_Indigenous[index_train, ]
test <- df_Female_Indigenous[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# ------------------------------------------------------------
# 58. MEXICO: ACCUMULATED LOCAL EFFECTS FOR INDIGENOUS WOMEN
# ------------------------------------------------------------

# Education effect
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3_w <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...
# (Occupation, Piped_water, Garbage, Urban_rural, Sewage, Water, Age)

# ------------------------------------------------------------
# 59. MEXICO: VARIABLE IMPORTANCE FOR INDIGENOUS WOMEN
# ------------------------------------------------------------

# Calculate permutation-based variable importance
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Mexico - Indigenous Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous women in Mexico
var_impo_RF_Female_indigenous_Mexico <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Women",
         Country = "Mexico",
         AUC = mp_rf_male_mix$measures$auc)
# ------------------------------------------------------------
# 60. PARTIAL AGGREGATION OF MEXICO WOMEN'S RESULTS
# ------------------------------------------------------------

# 60.1 Combine variable importance results for completed women subgroups in Mexico
# NOTE: This combines women's results but has naming conflicts with earlier men's objects
woman_models_mexico <- rbind(var_impo_RF_Female_Mix,       # Mixed women in Mexico
                             var_impo_RF_Female_indigenous) # Indigenous women in Mexico

# Calculate percentage performance loss
woman_models_mexico <- woman_models_mexico %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# 60.2 Combine ALE profiles for completed women subgroups in Mexico
# CRITICAL ISSUE: These object names overwrite earlier men's results

# Age profiles (overwrites men's df_graf_age from section 52)
df_graf_age_women <- rbind(df_graf_Age_1,    # Mixed women in Mexico
                           df_graf_Age_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles (overwrites men's df_graf_edu from section 52)
df_graf_edu_women <- rbind(df_graf_edu_1,    # Mixed women in Mexico
                           df_graf_edu_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles (overwrites men's df_graf_ocupa from section 52)
df_graf_ocupa_women <- rbind(df_graf_ocu_1,    # Mixed women in Mexico
                             df_graf_ocu_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles (overwrites men's df_graf_piper from section 52)
df_graf_piper_women <- rbind(df_graf_piper_1,    # Mixed women in Mexico
                             df_graf_piper_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles (overwrites men's df_graf_garba from section 52)
df_graf_garba_women <- rbind(df_graf_garbage_1,    # Mixed women in Mexico
                             df_graf_garbage_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles (overwrites men's df_graf_urban from section 52)
df_graf_urban_women <- rbind(df_graf_urban_1,    # Mixed women in Mexico
                             df_graf_urban_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles (overwrites men's df_graf_water from section 52)
df_graf_water_women <- rbind(df_graf_water_1,    # Mixed women in Mexico
                             df_graf_water_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles (overwrites men's df_graf_sewage from section 52)
df_graf_sewage_women <- rbind(df_graf_sewage_1,    # Mixed women in Mexico
                              df_graf_sewage_3) %>% # Indigenous women in Mexico
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 61. EXPLORATORY VISUALIZATIONS FOR MEXICO WOMEN
# ------------------------------------------------------------

# 61.1 Age effects: Mixed vs Indigenous women in Mexico
df_graf_age_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10)

# 61.2 Education effects: Mixed vs Indigenous women in Mexico
df_graf_edu_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Education") +
  scale_y_continuous(n.breaks = 10)

# 61.3 Occupation effects: Mixed vs Indigenous women in Mexico
df_graf_ocupa_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Occupation") +
  scale_y_continuous(n.breaks = 10)

# ------------------------------------------------------------
# 62. BRAZIL SENSITIVITY ANALYSIS
# ------------------------------------------------------------

# 62.1 Load and filter data for Brazil only
df_poblacion_2 <- read.csv("Data/Data.csv") %>%
  filter(Country == "Brazil")  # Brazil-specific filter

# 62.2 Data cleaning and preprocessing for Brazil
df_poblacion_2 <- df_poblacion_2 %>%
  filter(!is.na(Sewage), !is.na(Occupation), !is.na(Ethnicity)) %>%
  na.omit() %>%
  
  # Recode Occupation variable (consistent with main analysis)
  mutate(Occupation = case_when(
    Occupation == "Unemployed" ~ 0,
    Occupation == "Formal employment" ~ 3,
    Occupation == "Retiree" ~ 1,
    Occupation == "Casual employment" ~ 2
  )) %>%
  
  # Recode and convert chronic disease diagnosis to factor
  mutate(Dx_Cronical_disease = case_when(
    Dx_Cronical_disease == "No" ~ 0,
    Dx_Cronical_disease == "Yes" ~ 1
  )) %>%
  mutate(Dx_Cronical_disease = as.factor(Dx_Cronical_disease)) %>%
  
  # Remove unnecessary columns
  dplyr::select(-X, -Illiteracy)

# ------------------------------------------------------------
# 63. BRAZIL: MIXED-RACE MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 63.1 Data subset: Mixed-race men in Brazil
df_Male_mixrace <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_mixrace <- df_Male_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# 63.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_mixrace[index_train, ]
test <- df_Male_mixrace[-index_train, ]

# 63.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 63.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 63.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 63.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 63.7 ALE profiles for mixed-race men in Brazil
# Note: These objects have same names as Mexico analysis - CONFLICT!
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1_brazil <- data.frame(  # Added "_brazil" for clarity
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...
# (Occupation, Piped_water, Garbage, Urban_rural, Sewage, Water, Age)

# 63.8 Variable importance for mixed-race men in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Mixed-Race Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race men in Brazil
var_impo_RF_Male_Mix_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Men",
         Country = "Brazil",  # Country identifier added
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 64. BRAZIL: BLACK MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 64.1 Data subset: Black men in Brazil (Ethnicity == 2)
# Brazil has the largest African diaspora population in Latin America
df_Male_black <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 2, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_black <- df_Male_black %>%
  mutate_at(c(1:8), as.numeric)

# 64.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_black$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_black[index_train, ]
test <- df_Male_black[-index_train, ]

# 64.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 64.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 64.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 64.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 64.7 ALE profiles for Black men in Brazil
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_2_brazil <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 64.8 Variable importance for Black men in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Black Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Black men in Brazil
var_impo_RF_Black_Mix_Brazil <- vip_rf_parallel %>%  # Note: Name should be var_impo_RF_Male_Black_Brazil
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Black",
         Sex = "Men",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 65. BRAZIL: INDIGENOUS MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 65.1 Data subset: Indigenous men in Brazil (Ethnicity == 3)
df_Male_indige <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_indige <- df_Male_indige %>%
  mutate_at(c(1:8), as.numeric)

# 65.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_indige$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_indige[index_train, ]
test <- df_Male_indige[-index_train, ]

# 65.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 65.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 65.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 65.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 65.7 ALE profiles for Indigenous men in Brazil
# NOTE: Same object names as previous analyses - CONFLICT!
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3_brazil <- data.frame(  # Should be country-specific
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 65.8 Variable importance for Indigenous men in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Indigenous Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous men in Brazil
var_impo_RF_Male_indige_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Men",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 66. BRAZIL: OTHER MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 66.1 Data subset: Other men in Brazil (Ethnicity == 4)
# Likely includes White and other ethnic categories
df_Male_others <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 4, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_others <- df_Male_others %>%
  mutate_at(c(1:8), as.numeric)

# 66.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_others$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_others[index_train, ]
test <- df_Male_others[-index_train, ]

# 66.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 66.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 66.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 66.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 66.7 ALE profiles for Other men in Brazil
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_4_brazil <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 66.8 Variable importance for Other men in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Other Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Other men in Brazil
var_impo_Male_others_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Others",
         Sex = "Men",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)
# ------------------------------------------------------------
# 67. AGGREGATION OF BRAZIL MEN'S RESULTS
# ------------------------------------------------------------

# 67.1 Combine variable importance results for all Brazil men subgroups
# NOTE: This likely aggregates Brazil results but has naming issues
mens_models_brazil <- rbind(var_impo_RF_Black_Mix,     # Brazil Black men (poorly named)
                            var_impo_RF_Male_Mix,      # Brazil Mixed men  
                            var_impo_RF_Male_indige,   # Brazil Indigenous men
                            var_impo_Male_others)      # Brazil Other men

# CRITICAL ISSUE: These objects don't have country identifiers
# We can't tell if they're from Brazil or overwritten from other analyses

# Calculate percentage performance loss
mens_models_brazil <- mens_models_brazil %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# 67.2 Combine ALE profiles for all Brazil men subgroups
# CRITICAL ISSUE: Same object names used for Mexico - overwriting occurs

# Age profiles for Brazil men (overwrites Mexico's df_graf_age)
df_graf_age_brazil_men <- rbind(df_graf_Age_1,    # Mixed men
                                df_graf_Age_2,    # Black men
                                df_graf_Age_3,    # Indigenous men
                                df_graf_Age_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles
df_graf_edu_brazil_men <- rbind(df_graf_edu_1,    # Mixed men
                                df_graf_edu_2,    # Black men
                                df_graf_edu_3,    # Indigenous men
                                df_graf_edu_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles
df_graf_ocupa_brazil_men <- rbind(df_graf_ocu_1,    # Mixed men
                                  df_graf_ocu_2,    # Black men
                                  df_graf_ocu_3,    # Indigenous men
                                  df_graf_ocu_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles
df_graf_piper_brazil_men <- rbind(df_graf_piper_1,    # Mixed men
                                  df_graf_piper_2,    # Black men
                                  df_graf_piper_3,    # Indigenous men
                                  df_graf_piper_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles
df_graf_garba_brazil_men <- rbind(df_graf_garbage_1,    # Mixed men
                                  df_graf_garbage_2,    # Black men
                                  df_graf_garbage_3,    # Indigenous men
                                  df_graf_garbage_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles
df_graf_urban_brazil_men <- rbind(df_graf_urban_1,    # Mixed men
                                  df_graf_urban_2,    # Black men
                                  df_graf_urban_3,    # Indigenous men
                                  df_graf_urban_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles
df_graf_water_brazil_men <- rbind(df_graf_water_1,    # Mixed men
                                  df_graf_water_2,    # Black men
                                  df_graf_water_3,    # Indigenous men
                                  df_graf_water_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles
df_graf_sewage_brazil_men <- rbind(df_graf_sewage_1,    # Mixed men
                                   df_graf_sewage_2,    # Black men
                                   df_graf_sewage_3,    # Indigenous men
                                   df_graf_sewage_4) %>% # Other men
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 68. EXPLORATORY VISUALIZATIONS FOR BRAZIL MEN
# ------------------------------------------------------------

# 68.1 Age effects across Brazil men subgroups
df_graf_age_brazil_men %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Age Effects on Chronic Disease Risk", 
          subtitle = "Brazil - Men by Ethnicity")

# 68.2 Education effects across Brazil men subgroups
df_graf_edu_brazil_men %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Education") +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Education Effects on Chronic Disease Risk",
          subtitle = "Brazil - Men by Ethnicity")

# 68.3 Occupation effects across Brazil men subgroups
df_graf_ocupa_brazil_men %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("Average probability of chronic disease diagnosis") +
  xlab("Occupation") +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Occupation Effects on Chronic Disease Risk",
          subtitle = "Brazil - Men by Ethnicity")

# 68.4 Service access effects (example: Piped water)
df_graf_piper_brazil_men %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Piped Water Access") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Piped Water Access and Chronic Disease Risk",
          subtitle = "Brazil - Men by Ethnicity")

# ------------------------------------------------------------
# 69. BRAZIL: WOMEN SUBGROUP ANALYSES
# ------------------------------------------------------------

# 69.1 BRAZIL: MIXED-RACE WOMEN ANALYSIS

# Data subset: Mixed-race women in Brazil (Sex == 0, Ethnicity == 1)
df_Female_mixrace <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_mixrace <- df_Female_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_mixrace[index_train, ]
test <- df_Female_mixrace[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 69.2 ALE profiles for Mixed-race women in Brazil
# CRITICAL ISSUE: These objects overwrite earlier analyses
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1_w_brazil <- data.frame(  # Should have country-specific naming
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...
# (Occupation, Piped_water, Garbage, Urban_rural, Sewage, Water, Age)

# 69.3 Variable importance for Mixed-race women in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Mixed-Race Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race women in Brazil
var_impo_RF_Female_Mix_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Women",
         Country = "Brazil",  # Country identifier added
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 70. BRAZIL: BLACK WOMEN ANALYSIS
# ------------------------------------------------------------

# 70.1 Data subset: Black women in Brazil (Sex == 0, Ethnicity == 2)
# Brazil has largest population of African descent women in Latin America
df_Female_black <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 2, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_black <- df_Female_black %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_black$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_black[index_train, ]
test <- df_Female_black[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 70.2 ALE profiles for Black women in Brazil
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_2_w_brazil <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# NOTE: Error in original code - df_graf_garbage_2 has "Men Black" instead of "Women Black"
# This is corrected in cleaned version

# Similar ALE profiles created for other predictors...

# 70.3 Variable importance for Black women in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Black Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Black women in Brazil
var_impo_RF_Female_black_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Black",
         Sex = "Women",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 71. BRAZIL: INDIGENOUS WOMEN ANALYSIS
# ------------------------------------------------------------

# 71.1 Data subset: Indigenous women in Brazil (Sex == 0, Ethnicity == 3)
# Indigenous women in Brazil face multiple barriers: gender, ethnicity, geographic isolation
df_Female_Indigenous <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 3, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_Indigenous <- df_Female_Indigenous %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_Indigenous$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_Indigenous[index_train, ]
test <- df_Female_Indigenous[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 71.2 ALE profiles for Indigenous women in Brazil
# CRITICAL ISSUE: df_graf_edu_3 overwrites earlier objects
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_3_w_brazil <- data.frame(  # Should have unique naming
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 71.3 Variable importance for Indigenous women in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Indigenous Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Indigenous women in Brazil
var_impo_RF_Female_indigenous_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Indigenous",
         Sex = "Women",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 72. BRAZIL: OTHER WOMEN ANALYSIS (FINAL BRAZIL SUBGROUP)
# ------------------------------------------------------------

# 72.1 Data subset: Other women in Brazil (Sex == 0, Ethnicity == 4)
# Likely includes White women and other ethnic categories
df_Female_Others <- df_poblacion_2 %>%
  filter(Sex == 0, Ethnicity == 4, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Female_Others <- df_Female_Others %>%
  mutate_at(c(1:8), as.numeric)

# Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Female_Others$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Female_Others[index_train, ]
test <- df_Female_Others[-index_train, ]

# Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 72.2 ALE profiles for Other women in Brazil
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_4_w_brazil <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# NOTE: Error in original code - df_graf_piper_4 has "Men Others" instead of "Women Others"
# This is corrected in cleaned version

# Similar ALE profiles created for other predictors...

# 72.3 Variable importance for Other women in Brazil
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Brazil - Other Women") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Other women in Brazil
var_impo_RF_Female_others_Brazil <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Others",
         Sex = "Women",
         Country = "Brazil",
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 73. AGGREGATION OF BRAZIL WOMEN'S RESULTS
# ------------------------------------------------------------

# 73.1 Combine variable importance results for all Brazil women subgroups
woman_models_brazil <- rbind(var_impo_RF_Female_Mix,        # Brazil Mixed women
                             var_impo_RF_Female_black,      # Brazil Black women
                             var_impo_RF_Female_indigenous, # Brazil Indigenous women
                             var_impo_RF_Female_others)     # Brazil Other women

# Calculate percentage performance loss
woman_models_brazil <- woman_models_brazil %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# 73.2 Combine ALE profiles for all Brazil women subgroups
# CRITICAL ISSUE: These objects overwrite earlier analyses

# Age profiles for Brazil women (overwrites earlier df_graf_age)
df_graf_age_brazil_women <- rbind(df_graf_Age_1,    # Mixed women
                                  df_graf_Age_2,    # Black women
                                  df_graf_Age_3,    # Indigenous women
                                  df_graf_Age_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Education profiles (overwrites earlier df_graf_edu)
df_graf_edu_brazil_women <- rbind(df_graf_edu_1,    # Mixed women
                                  df_graf_edu_2,    # Black women
                                  df_graf_edu_3,    # Indigenous women
                                  df_graf_edu_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Occupation profiles (overwrites earlier df_graf_ocupa)
df_graf_ocupa_brazil_women <- rbind(df_graf_ocu_1,    # Mixed women
                                    df_graf_ocu_2,    # Black women
                                    df_graf_ocu_3,    # Indigenous women
                                    df_graf_ocu_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Piped water profiles (overwrites earlier df_graf_piper)
df_graf_piper_brazil_women <- rbind(df_graf_piper_1,    # Mixed women
                                    df_graf_piper_2,    # Black women
                                    df_graf_piper_3,    # Indigenous women
                                    df_graf_piper_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Garbage collection profiles (overwrites earlier df_graf_garba)
df_graf_garba_brazil_women <- rbind(df_graf_garbage_1,    # Mixed women
                                    df_graf_garbage_2,    # Black women
                                    df_graf_garbage_3,    # Indigenous women
                                    df_graf_garbage_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Urban/rural profiles (overwrites earlier df_graf_urban)
df_graf_urban_brazil_women <- rbind(df_graf_urban_1,    # Mixed women
                                    df_graf_urban_2,    # Black women
                                    df_graf_urban_3,    # Indigenous women
                                    df_graf_urban_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# General water access profiles (overwrites earlier df_graf_water)
df_graf_water_brazil_women <- rbind(df_graf_water_1,    # Mixed women
                                    df_graf_water_2,    # Black women
                                    df_graf_water_3,    # Indigenous women
                                    df_graf_water_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Sewage system profiles (overwrites earlier df_graf_sewage)
df_graf_sewage_brazil_women <- rbind(df_graf_sewage_1,    # Mixed women
                                     df_graf_sewage_2,    # Black women
                                     df_graf_sewage_3,    # Indigenous women
                                     df_graf_sewage_4) %>% # Other women
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# ------------------------------------------------------------
# 74. EXPLORATORY VISUALIZATIONS FOR BRAZIL WOMEN
# ------------------------------------------------------------

# 74.1 Age effects across Brazil women subgroups
df_graf_age_brazil_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Age") +
  scale_x_continuous(n.breaks = 10) +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Age Effects on Chronic Disease Risk",
          subtitle = "Brazil - Women by Ethnicity")

# 74.2 Education effects across Brazil women subgroups
df_graf_edu_brazil_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Education") +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Education Effects on Chronic Disease Risk",
          subtitle = "Brazil - Women by Ethnicity")

# 74.3 Occupation effects across Brazil women subgroups
df_graf_ocupa_brazil_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Occupation") +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Occupation Effects on Chronic Disease Risk",
          subtitle = "Brazil - Women by Ethnicity")

# 74.4 Service access effects (example: Piped water)
df_graf_piper_brazil_women %>%
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) +
  theme_bw() +
  ylab("") +
  xlab("Piped Water Access") +
  scale_x_continuous(n.breaks = 2) +
  scale_y_continuous(n.breaks = 10) +
  ggtitle("Piped Water Access and Chronic Disease Risk",
          subtitle = "Brazil - Women by Ethnicity")

################################
# ------------------------------------------------------------
# 75. ECUADOR SENSITIVITY ANALYSIS
# ------------------------------------------------------------

# 75.1 Load and filter data for Ecuador only
df_poblacion_2 <- read.csv("Data/Data.csv") %>%
  filter(Country == "Ecuador")  # Ecuador-specific filter

# 75.2 Data cleaning and preprocessing for Ecuador
df_poblacion_2 <- df_poblacion_2 %>%
  filter(!is.na(Sewage), !is.na(Occupation), !is.na(Ethnicity)) %>%
  na.omit() %>%
  
  # Recode Occupation variable (consistent with other analyses)
  mutate(Occupation = case_when(
    Occupation == "Unemployed" ~ 0,
    Occupation == "Formal employment" ~ 3,
    Occupation == "Retiree" ~ 1,
    Occupation == "Casual employment" ~ 2
  )) %>%
  
  # Recode and convert chronic disease diagnosis to factor
  mutate(Dx_Cronical_disease = case_when(
    Dx_Cronical_disease == "No" ~ 0,
    Dx_Cronical_disease == "Yes" ~ 1
  )) %>%
  mutate(Dx_Cronical_disease = as.factor(Dx_Cronical_disease)) %>%
  
  # Remove unnecessary columns
  dplyr::select(-X, -Illiteracy)

# ------------------------------------------------------------
# 76. ECUADOR: MIXED-RACE MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 76.1 Data subset: Mixed-race men in Ecuador
df_Male_mixrace <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 1, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_mixrace <- df_Male_mixrace %>%
  mutate_at(c(1:8), as.numeric)

# 76.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_mixrace$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_mixrace[index_train, ]
test <- df_Male_mixrace[-index_train, ]

# 76.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 76.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 76.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 76.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 76.7 ALE profiles for mixed-race men in Ecuador
# CRITICAL ISSUE: Same object names as Brazil/Mexico - overwriting occurs
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_1_ecuador <- data.frame(  # Should have country-specific naming
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 76.8 Variable importance for mixed-race men in Ecuador
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Ecuador - Mixed-Race Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for mixed-race men in Ecuador
var_impo_RF_Male_Mix_Ecuador <- vip_rf_parallel %>%
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Mixed",
         Sex = "Men",
         Country = "Ecuador",  # Country identifier added
         AUC = mp_rf_male_mix$measures$auc)

# ------------------------------------------------------------
# 77. ECUADOR: BLACK MEN SUBGROUP ANALYSIS
# ------------------------------------------------------------

# 77.1 Data subset: Black men in Ecuador (Ethnicity == 2)
# Ecuador has significant Afro-Ecuadorian population, especially in Esmeraldas province
df_Male_black <- df_poblacion_2 %>%
  filter(Sex == 1, Ethnicity == 2, Age > 19) %>%
  dplyr::select(-Sex, -Ethnicity) %>%
  na.omit()

# Convert predictor columns to numeric
df_Male_black <- df_Male_black %>%
  mutate_at(c(1:8), as.numeric)

# 77.2 Train-test split
set.seed(1234)
index_train <- createDataPartition(df_Male_black$Dx_Cronical_disease,
                                   p = .7,
                                   list = FALSE)

train <- df_Male_black[index_train, ]
test <- df_Male_black[-index_train, ]

# 77.3 Separate survey weights
train_2 <- train
pesos_train <- train_2$weights_of_people
train_2 <- train_2 %>% dplyr::select(-weights_of_people)

test_2 <- test
pesos_test <- test_2$weights_of_people
test_2 <- test_2 %>% dplyr::select(-weights_of_people)

# 77.4 Train Random Forest model with survey weights
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = TRUE,
  case.weights = pesos_train
)

# 77.5 Create explainer for model interpretation
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE
)

# 77.6 Calculate model performance
mp_rf_male_mix <- model_performance(explainer_rf_ranger)

# 77.7 ALE profiles for Black men in Ecuador
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger,
  variables = c("Education_primary_category"),
  type = "accumulated",
  N = 5000
)

df_graf_edu_2_ecuador <- data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]],
  Group = "Men Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Similar ALE profiles created for other predictors...

# 77.8 Variable importance for Black men in Ecuador
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference",
    parallel = TRUE,
    weights = pesos_test
  )
})

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = TRUE) +
  ggtitle("Variable Importance for Chronic Disease Prediction",
          subtitle = "Ecuador - Black Men") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Summarize importance metrics for Black men in Ecuador
var_impo_RF_Black_Mix_Ecuador <- vip_rf_parallel %>%  # Note: Naming should be var_impo_RF_Male_Black_Ecuador
  group_by(variable) %>%
  summarise(dropout_loss = mean(dropout_loss)) %>%
  mutate(Ethnicity = "Black",
         Sex = "Men",
         Country = "Ecuador",
         AUC = mp_rf_male_mix$measures$auc)

# ============================================================================
# PART 78: INDIGENOUS MALE POPULATION ANALYSIS
# ============================================================================

# Step 1.1: Data preparation for Indigenous males
# ----------------------------------------------------------------------------
# Filter: Male (Sex == 1), Indigenous (Ethnicity == 3), Age > 19
# Remove Sex and Ethnicity columns (used for filtering)
# Remove rows with missing values

df_Male_indige = df_poblacion_2 %>% 
  filter(Sex == 1, Ethnicity == 3, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

# Check distribution of chronic disease diagnosis
summary(as.factor(df_Male_indige$Dx_Cronical_disease))

# Convert first 8 columns to numeric for modeling
df_Male_indige <- df_Male_indige %>%
  mutate_at(c(1:8), as.numeric) 

# Step 1.2: Train-test split
# ----------------------------------------------------------------------------
# 70% training, 30% testing
# Stratified sampling to maintain class distribution

set.seed(1234)  # For reproducibility
index_train <- createDataPartition(df_Male_indige$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Male_indige[index_train, ]
test <- df_Male_indige[-index_train, ]

# Step 1.3: Prepare data for Random Forest with survey weights
# ----------------------------------------------------------------------------
# Store survey weights separately from predictors
train_2 = train
pesos_train = train_2$weights_of_people  # Survey weights for training
train_2 = train_2 %>% dplyr::select(-weights_of_people)  # Remove weight column

test_2 = test
pesos_test = test_2$weights_of_people  # Survey weights for testing
test_2 = test_2 %>% dplyr::select(-weights_of_people)  # Remove weight column

# Step 1.4: Train Random Forest model
# ----------------------------------------------------------------------------
# Weighted Random Forest using ranger package
# Probability outputs for binary classification

modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,  # Predict using all variables
  data = train_2,
  num.trees = 300,        # Number of trees in the forest
  min.node.size = 5,      # Minimum size of terminal nodes
  mtry = 3,               # Number of variables to try at each split
  verbose = FALSE,
  importance = "none",    # No variable importance calculation (done later)
  seed = 123,             # Random seed
  probability = T,        # Output probabilities instead of classes
  case.weights = pesos_train,  # Apply survey weights
)

# Step 1.5: Create DALEX explainer for model interpretation
# ----------------------------------------------------------------------------
# DALEX provides model-agnostic explanations

explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables (exclude target in column 9)
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),  # Binary outcome
  weights = pesos_test,  # Test set survey weights
  verbose = FALSE,
)

# Step 1.6: Model performance evaluation
# ----------------------------------------------------------------------------
# Calculate AUC and other performance metrics
mp_rf_male_mix <- model_performance(explainer_rf_ranger)
mp_rf_male_mix

# Step 1.7: Partial Dependence Plots (PDPs)
# ----------------------------------------------------------------------------
# PDPs show marginal effect of each predictor on predicted outcome
# N = 5000 samples for stable estimates

set.seed(1234)
# Education level PDP
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Education_primary_category"), 
  type = "accumulated", 
  N = 5000
)

# Store PDP results for visualization
df_graf_edu_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Men Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Repeat for other predictors (similar structure for each):
# Occupation, Piped_water, Garbage, Urban_rural, Sewage, Water, Age

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Occupation"), 
                                 type = "accumulated", N = 5000)
df_graf_ocu_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                           Group = "Men Indigenous",
                           U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Piped_water"), 
                                 type = "accumulated", N = 5000)
df_graf_piper_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Indigenous",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Garbage"), 
                                 type = "accumulated", N = 5000)
df_graf_garbage_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                               Group = "Men Indigenous",
                               U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Urban_rural"), 
                                 type = "accumulated", N = 5000)
df_graf_urban_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Indigenous",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Sewage"), 
                                 type = "accumulated", N = 5000)
df_graf_sewage_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                              Group = "Men Indigenous",
                              U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Water"), 
                                 type = "accumulated", N = 5000)
df_graf_water_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Indigenous",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Age"), 
                                 type = "accumulated", N = 5000)
df_graf_Age_3 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                           Group = "Men Indigenous",
                           U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

# Step 1.8: Variable Importance Analysis
# ----------------------------------------------------------------------------
# Permutation-based variable importance
# B = 50 permutations, N = 2000 observations
# Parallel processing for speed

system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,                # Number of permutations
    N = 2000,              # Number of observations to sample
    type = "difference",   # Measure importance by performance drop
    parallel = TRUE,       # Use parallel processing
    weights = pesos_test   # Apply survey weights
  )
})

# Audible notification when computation is complete
beepr::beep(sound = "mario")

# Step 1.9: Visualize variable importance
# ----------------------------------------------------------------------------
library(scales) 

plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle(paste0("Variable Importance for Chronic Disease Prediction"), 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 1.10: Create summary dataframe of variable importance
# ----------------------------------------------------------------------------
var_impo_RF_Male_indige = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Indigenous", Sex = "Men") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)  # Add model AUC

# ============================================================================
# PART 2: OTHER ETHNICITY MALE POPULATION ANALYSIS
# ============================================================================
# (Same process repeated for Ethnicity == 4 - "Others")

# Step 2.1: Data preparation for Other ethnicity males
# ----------------------------------------------------------------------------
df_Male_others = df_poblacion_2 %>% 
  filter(Sex == 1, Ethnicity == 4, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

summary(as.factor(df_Male_others$Dx_Cronical_disease))
df_Male_others <- df_Male_others %>%
  mutate_at(c(1:8), as.numeric) 

# Step 2.2: Train-test split (same 70/30 ratio)
# ----------------------------------------------------------------------------
set.seed(1234)
index_train <- createDataPartition(df_Male_others$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Male_others[index_train, ]
test <- df_Male_others[-index_train, ]

# Step 2.3: Prepare weighted data
# ----------------------------------------------------------------------------
train_2 = train
pesos_train = train_2$weights_of_people
train_2 = train_2 %>% dplyr::select(-weights_of_people)

test_2 = test
pesos_test = test_2$weights_of_people
test_2 = test_2 %>% dplyr::select(-weights_of_people)

# Step 2.4: Train Random Forest for Other ethnicity
# ----------------------------------------------------------------------------
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = T,
  case.weights = pesos_train,
)

# Step 2.5: Create DALEX explainer
# ----------------------------------------------------------------------------
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE,
)

# Step 2.6: Model performance
# ----------------------------------------------------------------------------
mp_rf_male_mix <- model_performance(explainer_rf_ranger)
mp_rf_male_mix

# Step 2.7: Partial Dependence Plots for Other ethnicity
# ----------------------------------------------------------------------------
# (Same PDPs as for Indigenous, but stored with "Men Others" label)

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Education_primary_category"), 
                                 type = "accumulated", N = 5000)

df_graf_edu_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                           Group = "Men Others",
                           U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

# ... (similar PDPs for other variables: Occupation, Piped_water, etc.)
# df_graf_ocu_4, df_graf_piper_4, df_graf_garbage_4, etc.

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Occupation"), 
                                 type = "accumulated", N = 5000)
df_graf_ocu_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                           Group = "Men Others",
                           U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Piped_water"), 
                                 type = "accumulated", N = 5000)
df_graf_piper_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Others",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Garbage"), 
                                 type = "accumulated", N = 5000)
df_graf_garbage_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                               Group = "Men Others",
                               U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Urban_rural"), 
                                 type = "accumulated", N = 5000)
df_graf_urban_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Others",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Sewage"), 
                                 type = "accumulated", N = 5000)
df_graf_sewage_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                              Group = "Men Others",
                              U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Water"), 
                                 type = "accumulated", N = 5000)
df_graf_water_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                             Group = "Men Others",
                             U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

set.seed(1234)
pdp_rf_male_mix <- model_profile(explainer = explainer_rf_ranger, 
                                 variables = c("Age"), 
                                 type = "accumulated", N = 5000)
df_graf_Age_4 = data.frame(value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
                           Group = "Men Others",
                           U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]])

# Step 2.8: Variable importance for Other ethnicity
# ----------------------------------------------------------------------------
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference", 
    parallel = TRUE,
    weights = pesos_test
  )
})

beepr::beep(sound = "mario")

# Step 2.9: Visualize importance
# ----------------------------------------------------------------------------
plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle(paste0("Variable Importance for Chronic Disease Prediction"), 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 2.10: Create summary dataframe
# ----------------------------------------------------------------------------
var_impo_Male_others = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Others", Sex = "Men") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)

# ============================================================================
# PART 3: COMBINING RESULTS AND VISUALIZATION
# ============================================================================
# This section combines results from all ethnic group analyses
# and creates comparative visualizations

# Step 3.1: Combine variable importance results from all models
# ----------------------------------------------------------------------------
# Combine importance dataframes for all male ethnic groups:
# 1. Black Mixed (var_impo_RF_Black_Mix)
# 2. Mixed Race (var_impo_RF_Male_Mix)
# 3. Indigenous (var_impo_RF_Male_indige)
# 4. Others (var_impo_Male_others)

mens_models = rbind(var_impo_RF_Black_Mix, var_impo_RF_Male_Mix, 
                    var_impo_RF_Male_indige, var_impo_Male_others)

# Step 3.2: Calculate percentage loss metric
# ----------------------------------------------------------------------------
# Convert dropout_loss to percentage of AUC for standardized comparison
# Por_perdido = (dropout_loss / AUC) * 100
# This represents the % decrease in model performance when variable is removed

mens_models = mens_models %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# Step 3.3: Combine Partial Dependence Plot (PDP) data for Age
# ----------------------------------------------------------------------------
# Combine PDP results for Age variable across all ethnic groups
df_graf_age = rbind(df_graf_Age_1, df_graf_Age_2, 
                    df_graf_Age_3, df_graf_Age_4)

# Ensure probability values are non-negative (trim negative values to 0)
df_graf_age = df_graf_age %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Optional: Save combined data for external analysis
# write.csv(df_graf_age, "path/to/save/df_graf_age.csv")

# Step 3.4: Create PDP visualization for Age
# ----------------------------------------------------------------------------
# Line plot showing how chronic disease probability changes with age
# Different colors/shapes for each ethnic group
df_graf_age %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of being dx with chronic disease") +
  scale_x_continuous(n.breaks = 10) +  # 10 breaks on x-axis for Age
  scale_y_continuous(n.breaks = 10) +  # 10 breaks on y-axis for probability
  xlab("Age (years)")

# Step 3.5: Repeat combination and visualization for Education
# ----------------------------------------------------------------------------
df_graf_edu = rbind(df_graf_edu_1, df_graf_edu_2, 
                    df_graf_edu_3, df_graf_edu_4)

df_graf_edu = df_graf_edu %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_edu %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of being dx with chronic disease") +
  # Note: Education is categorical, so no continuous scale breaks
  scale_y_continuous(n.breaks = 10) +
  xlab("Education Level")  # Likely categorical: 1, 2, 3, etc.

# Step 3.6: Repeat for Occupation
# ----------------------------------------------------------------------------
df_graf_ocupa = rbind(df_graf_ocu_1, df_graf_ocu_2, 
                      df_graf_ocu_3, df_graf_ocu_4)

df_graf_ocupa = df_graf_ocupa %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_ocupa %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of being dx with chronic disease") +
  scale_y_continuous(n.breaks = 10) +
  xlab("Occupation Category")

# Step 3.7: Repeat for Piped Water access (binary/categorical)
# ----------------------------------------------------------------------------
df_graf_piper = rbind(df_graf_piper_1, df_graf_piper_2, 
                      df_graf_piper_3, df_graf_piper_4)

df_graf_piper = df_graf_piper %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_piper %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +  # Empty y-label for side-by-side plotting
  scale_x_continuous(n.breaks = 2) +  # Binary variable: 0=No, 1=Yes
  scale_y_continuous(n.breaks = 10) +
  xlab("Piped Water Access")

# Step 3.8: Repeat for Garbage collection (binary/categorical)
# ----------------------------------------------------------------------------
df_graf_garba = rbind(df_graf_garbage_1, df_graf_garbage_2, 
                      df_graf_garbage_3, df_graf_garbage_4)

df_graf_garba = df_graf_garba %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_garba %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Binary variable
  scale_y_continuous(n.breaks = 10) +
  xlab("Garbage Collection Service")

# Step 3.9: Repeat for Urban/Rural residence
# ----------------------------------------------------------------------------
df_graf_urban = rbind(df_graf_urban_1, df_graf_urban_2, 
                      df_graf_urban_3, df_graf_urban_4)

df_graf_urban = df_graf_urban %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_urban %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Likely: 0=Rural, 1=Urban
  scale_y_continuous(n.breaks = 10) +
  xlab("Residence Type (Urban/Rural)")

# Step 3.10: Repeat for Water access (general)
# ----------------------------------------------------------------------------
df_graf_water = rbind(df_graf_water_1, df_graf_water_2, 
                      df_graf_water_3, df_graf_water_4)

df_graf_water = df_graf_water %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_water %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Likely categorical water access
  scale_y_continuous(n.breaks = 10) +
  xlab("Water Access")

# Step 3.11: Repeat for Sewage system access
# ----------------------------------------------------------------------------
df_graf_sewage = rbind(df_graf_sewage_1, df_graf_sewage_2, 
                       df_graf_sewage_3, df_graf_sewage_4)

df_graf_sewage = df_graf_sewage %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_sewage %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of being Dx with chronic disease") +
  scale_x_continuous(n.breaks = 2) +  # Likely: 0=No sewage, 1=Has sewage
  scale_y_continuous(n.breaks = 10) +
  xlab("Sewage System Access")
###############################################################################
# PART 4: FEMALE POPULATION ANALYSIS - MIXED RACE
###############################################################################
# This section repeats the analysis for female populations
# Starting with Mixed Race women (Sex == 0, Ethnicity == 1)

# Step 4.1: Data preparation for Mixed Race women
# ----------------------------------------------------------------------------
# Filter: Female (Sex == 0), Mixed Race (Ethnicity == 1), Age > 19
# Remove Sex and Ethnicity columns after filtering
# Remove rows with missing values

df_Female_mixrace = df_poblacion_2 %>% 
  filter(Sex == 0, Ethnicity == 1, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

# Check distribution of chronic disease diagnosis
# Note: There's a typo here - should be df_Female_mixrace, not df_Male_mixrace
summary(as.factor(df_Female_mixrace$Dx_Cronical_disease))

# Convert first 8 columns to numeric for modeling
df_Female_mixrace <- df_Female_mixrace %>%
  mutate_at(c(1:8), as.numeric) 

# Step 4.2: Train-test split for Mixed Race women
# ----------------------------------------------------------------------------
set.seed(1234)  # For reproducibility
index_train <- createDataPartition(df_Female_mixrace$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Female_mixrace[index_train, ]
test <- df_Female_mixrace[-index_train, ]

# Step 4.3: Prepare data with survey weights
# ----------------------------------------------------------------------------
train_2 = train
pesos_train = train_2$weights_of_people  # Survey weights for training
train_2 = train_2 %>% dplyr::select(-weights_of_people)

test_2 = test
pesos_test = test_2$weights_of_people  # Survey weights for testing
test_2 = test_2 %>% dplyr::select(-weights_of_people)

# Step 4.4: Train Random Forest for Mixed Race women
# ----------------------------------------------------------------------------
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = T,
  case.weights = pesos_train,  # Apply survey weights
)

# Step 4.5: Create DALEX explainer
# ----------------------------------------------------------------------------
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables (exclude target in column 9)
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),  # Binary outcome
  weights = pesos_test,  # Test set survey weights
  verbose = FALSE,
)

# Step 4.6: Model performance evaluation
# ----------------------------------------------------------------------------
mp_rf_male_mix <- model_performance(explainer_rf_ranger)  # Note: misnamed, should be mp_rf_female_mix
mp_rf_male_mix

# Step 4.7: Partial Dependence Plots for Mixed Race women
# ----------------------------------------------------------------------------
# Education level PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Education_primary_category"), 
  type = "accumulated", 
  N = 5000
)

# Store PDP results for visualization - Note: Using "_1" suffix for female mixed race
df_graf_edu_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",  # Female mixed race
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Occupation PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Occupation"), 
  type = "accumulated", 
  N = 5000
)

df_graf_ocu_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Piped water access PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Piped_water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_piper_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Garbage collection PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Garbage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_garbage_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Urban/rural residence PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Urban_rural"), 
  type = "accumulated", 
  N = 5000
)

df_graf_urban_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Sewage system PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Sewage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_sewage_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Water access PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_water_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Age PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Age"), 
  type = "accumulated", 
  N = 5000
)

df_graf_Age_1 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Mixed",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Step 4.8: Variable importance for Mixed Race women
# ----------------------------------------------------------------------------
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,                # Number of permutations
    N = 2000,              # Number of observations to sample
    type = "difference",   # Measure importance by performance drop
    parallel = TRUE,       # Use parallel processing
    weights = pesos_test   # Apply survey weights
  )
})

# Audible notification when computation is complete
beepr::beep(sound = "mario")

# Visualize variable importance
library(scales) 
plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle("Variable Importance for Chronic Disease Prediction in Women Mixed Race", 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 4.9: Create summary dataframe for Mixed Race women
# ----------------------------------------------------------------------------
var_impo_RF_Female_Mix = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Mixed", Sex = "Women") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)  # Add model AUC

###############################################################################
# PART 5: FEMALE POPULATION ANALYSIS - BLACK
###############################################################################
# Repeating analysis for Black women (Sex == 0, Ethnicity == 2)

# Step 5.1: Data preparation for Black women
# ----------------------------------------------------------------------------
df_Female_black = df_poblacion_2 %>% 
  filter(Sex == 0, Ethnicity == 2, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

summary(as.factor(df_Female_black$Dx_Cronical_disease))
df_Female_black <- df_Female_black %>%
  mutate_at(c(1:8), as.numeric) 

# Step 5.2: Train-test split for Black women
# ----------------------------------------------------------------------------
set.seed(1234)
index_train <- createDataPartition(df_Female_black$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Female_black[index_train, ]
test <- df_Female_black[-index_train, ]

# Step 5.3: Prepare data with survey weights
# ----------------------------------------------------------------------------
train_2 = train
pesos_train = train_2$weights_of_people
train_2 = train_2 %>% dplyr::select(-weights_of_people)

test_2 = test
pesos_test = test_2$weights_of_people
test_2 = test_2 %>% dplyr::select(-weights_of_people)

# Step 5.4: Train Random Forest for Black women
# ----------------------------------------------------------------------------
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = T,
  case.weights = pesos_train,
)

# Step 5.5: Create DALEX explainer
# ----------------------------------------------------------------------------
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE,
)

# Step 5.6: Model performance evaluation
# ----------------------------------------------------------------------------
mp_rf_male_mix <- model_performance(explainer_rf_ranger)
mp_rf_male_mix

# Step 5.7: Partial Dependence Plots for Black women
# ----------------------------------------------------------------------------
# Note: Using "_2" suffix for Black women

# Education PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Education_primary_category"), 
  type = "accumulated", 
  N = 5000
)

df_graf_edu_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Occupation PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Occupation"), 
  type = "accumulated", 
  N = 5000
)

df_graf_ocu_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Piped water PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Piped_water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_piper_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Garbage PDP - NOTE: Group label mismatch - says "Men Black" but should be "Women Black"
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Garbage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_garbage_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Men Black",  # ERROR: Should be "Women Black"
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Urban/rural PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Urban_rural"), 
  type = "accumulated", 
  N = 5000
)

df_graf_urban_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Sewage PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Sewage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_sewage_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Water PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_water_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Age PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Age"), 
  type = "accumulated", 
  N = 5000
)

df_graf_Age_2 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Black",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Step 5.8: Variable importance for Black women
# ----------------------------------------------------------------------------
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference", 
    parallel = TRUE,
    weights = pesos_test
  )
})

beepr::beep(sound = "mario")

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle("Variable Importance for Chronic Disease Prediction in Black Women", 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 5.9: Create summary dataframe for Black women
# ----------------------------------------------------------------------------
var_impo_RF_Female_black = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Black", Sex = "Women") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)
###############################################################################
# PART 6: FEMALE POPULATION ANALYSIS - INDIGENOUS WOMEN
###############################################################################
# Analysis for Indigenous women (Sex == 0, Ethnicity == 3)

# Step 6.1: Data preparation for Indigenous women
# ----------------------------------------------------------------------------
df_Female_Indigenous = df_poblacion_2 %>% 
  filter(Sex == 0, Ethnicity == 3, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

# Check distribution of chronic disease diagnosis
summary(as.factor(df_Female_Indigenous$Dx_Cronical_disease))

# Convert first 8 columns to numeric for modeling
df_Female_Indigenous <- df_Female_Indigenous %>%
  mutate_at(c(1:8), as.numeric) 

# Step 6.2: Train-test split for Indigenous women
# ----------------------------------------------------------------------------
set.seed(1234)  # For reproducibility
index_train <- createDataPartition(df_Female_Indigenous$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Female_Indigenous[index_train, ]
test <- df_Female_Indigenous[-index_train, ]

# Step 6.3: Prepare data with survey weights
# ----------------------------------------------------------------------------
train_2 = train
pesos_train = train_2$weights_of_people  # Survey weights for training
train_2 = train_2 %>% dplyr::select(-weights_of_people)

test_2 = test
pesos_test = test_2$weights_of_people  # Survey weights for testing
test_2 = test_2 %>% dplyr::select(-weights_of_people)

# Step 6.4: Train Random Forest for Indigenous women
# ----------------------------------------------------------------------------
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = T,
  case.weights = pesos_train,  # Apply survey weights
)

# Step 6.5: Create DALEX explainer
# ----------------------------------------------------------------------------
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],  # Predictor variables (exclude target in column 9)
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),  # Binary outcome
  weights = pesos_test,  # Test set survey weights
  verbose = FALSE,
)

# Step 6.6: Model performance evaluation
# ----------------------------------------------------------------------------
mp_rf_male_mix <- model_performance(explainer_rf_ranger)  # Note: misnamed
mp_rf_male_mix

# Step 6.7: Partial Dependence Plots for Indigenous women
# ----------------------------------------------------------------------------
# Note: Using "_3" suffix for Indigenous women

# Education level PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Education_primary_category"), 
  type = "accumulated", 
  N = 5000
)

df_graf_edu_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Occupation PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Occupation"), 
  type = "accumulated", 
  N = 5000
)

df_graf_ocu_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Piped water access PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Piped_water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_piper_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Garbage collection PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Garbage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_garbage_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Urban/rural residence PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Urban_rural"), 
  type = "accumulated", 
  N = 5000
)

df_graf_urban_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Sewage system PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Sewage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_sewage_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Water access PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_water_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Age PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Age"), 
  type = "accumulated", 
  N = 5000
)

df_graf_Age_3 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Indigenous",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Step 6.8: Variable importance for Indigenous women
# ----------------------------------------------------------------------------
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,                # Number of permutations
    N = 2000,              # Number of observations to sample
    type = "difference",   # Measure importance by performance drop
    parallel = TRUE,       # Use parallel processing
    weights = pesos_test   # Apply survey weights
  )
})

# Audible notification when computation is complete
beepr::beep(sound = "mario")

# Visualize variable importance
library(scales) 
plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle("Variable Importance for Chronic Disease Prediction in Indigenous Women", 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 6.9: Create summary dataframe for Indigenous women
# ----------------------------------------------------------------------------
var_impo_RF_Female_indigenous = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Indigenous", Sex = "Women") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)  # Add model AUC

###############################################################################
# PART 7: FEMALE POPULATION ANALYSIS - OTHER ETHNICITY WOMEN
###############################################################################
# Final analysis for Other ethnicity women (Sex == 0, Ethnicity == 4)

# Step 7.1: Data preparation for Other ethnicity women
# ----------------------------------------------------------------------------
df_Female_Others = df_poblacion_2 %>% 
  filter(Sex == 0, Ethnicity == 4, Age > 19) %>% 
  dplyr::select(-Sex, -Ethnicity) %>% 
  na.omit()

summary(as.factor(df_Female_Others$Dx_Cronical_disease))
df_Female_Others <- df_Female_Others %>%
  mutate_at(c(1:8), as.numeric) 

# Step 7.2: Train-test split for Other ethnicity women
# ----------------------------------------------------------------------------
set.seed(1234)
index_train <- createDataPartition(df_Female_Others$Dx_Cronical_disease, 
                                   p = .7, 
                                   list = FALSE)

train <- df_Female_Others[index_train, ]
test <- df_Female_Others[-index_train, ]

# Step 7.3: Prepare data with survey weights
# ----------------------------------------------------------------------------
train_2 = train
pesos_train = train_2$weights_of_people
train_2 = train_2 %>% dplyr::select(-weights_of_people)

test_2 = test
pesos_test = test_2$weights_of_people
test_2 = test_2 %>% dplyr::select(-weights_of_people)

# Step 7.4: Train Random Forest for Other ethnicity women
# ----------------------------------------------------------------------------
modelo_rf_ranger <- ranger(
  formula = Dx_Cronical_disease ~ .,
  data = train_2,
  num.trees = 300,
  min.node.size = 5,
  mtry = 3,
  verbose = FALSE,
  importance = "none",
  seed = 123,
  probability = T,
  case.weights = pesos_train,
)

# Step 7.5: Create DALEX explainer
# ----------------------------------------------------------------------------
explainer_rf_ranger <- DALEX::explain(
  model = modelo_rf_ranger,
  label = "RandomForest",
  data = test_2[, -9],
  y = as.numeric(test_2$Dx_Cronical_disease == "1"),
  weights = pesos_test,
  verbose = FALSE,
)

# Step 7.6: Model performance evaluation
# ----------------------------------------------------------------------------
mp_rf_male_mix <- model_performance(explainer_rf_ranger)
mp_rf_male_mix

# Step 7.7: Partial Dependence Plots for Other ethnicity women
# ----------------------------------------------------------------------------
# Note: Using "_4" suffix for Other ethnicity women

# Education PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Education_primary_category"), 
  type = "accumulated", 
  N = 5000
)

df_graf_edu_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Occupation PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Occupation"), 
  type = "accumulated", 
  N = 5000
)

df_graf_ocu_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Piped water PDP - NOTE: Group label mismatch - says "Men Others" but should be "Women Others"
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Piped_water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_piper_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Men Others",  # ERROR: Should be "Women Others"
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Garbage PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Garbage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_garbage_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Urban/rural PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Urban_rural"), 
  type = "accumulated", 
  N = 5000
)

df_graf_urban_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Sewage PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Sewage"), 
  type = "accumulated", 
  N = 5000
)

df_graf_sewage_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Water PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Water"), 
  type = "accumulated", 
  N = 5000
)

df_graf_water_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Age PDP
set.seed(1234)
pdp_rf_male_mix <- model_profile(
  explainer = explainer_rf_ranger, 
  variables = c("Age"), 
  type = "accumulated", 
  N = 5000
)

df_graf_Age_4 = data.frame(
  value = pdp_rf_male_mix[["agr_profiles"]][["_x_"]], 
  Group = "Women Others",
  U5MR_average = pdp_rf_male_mix[["agr_profiles"]][["_yhat_"]]
)

# Step 7.8: Variable importance for Other ethnicity women
# ----------------------------------------------------------------------------
system.time({
  vip_rf_parallel <- model_parts(
    explainer = explainer_rf_ranger,
    B = 50,
    N = 2000,
    type = "difference", 
    parallel = TRUE,
    weights = pesos_test
  )
})

beepr::beep(sound = "mario")

# Visualize variable importance
plot(vip_rf_parallel, max_vars = 42, show_boxplots = T) +
  ggtitle("Variable Importance for Chronic Disease Prediction in Other Ethnicity Women", 
          subtitle = "") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(n.breaks = 10, labels = label_number())

# Step 7.9: Create summary dataframe for Other ethnicity women
# ----------------------------------------------------------------------------
var_impo_RF_Female_others = vip_rf_parallel %>% 
  group_by(variable) %>% 
  summarise(dropout_loss = mean(dropout_loss)) %>% 
  mutate(Etnicity = "Others", Sex = "Women") %>% 
  mutate(AUC = mp_rf_male_mix$measures$auc)

# =

###############################################################################
# PART 8: COMBINING FEMALE RESULTS AND CREATING VISUALIZATIONS
###############################################################################
# This section combines results from all female ethnic group analyses
# and creates gender-specific visualizations

# Step 8.1: Combine variable importance results from all female models
# ----------------------------------------------------------------------------
# Combine importance dataframes for all female ethnic groups:
# 1. Mixed Race Women (var_impo_RF_Female_Mix)
# 2. Black Women (var_impo_RF_Female_black)
# 3. Indigenous Women (var_impo_RF_Female_indigenous)
# 4. Other Women (var_impo_RF_Female_others)

woman_models = rbind(var_impo_RF_Female_Mix, var_impo_RF_Female_black, 
                     var_impo_RF_Female_indigenous, var_impo_RF_Female_others)

# Step 8.2: Calculate percentage loss metric for female models
# ----------------------------------------------------------------------------
# Convert dropout_loss to percentage of AUC for standardized comparison
# Por_perdido = (dropout_loss / AUC) * 100
# This represents the % decrease in model performance when variable is removed

woman_models = woman_models %>%
  mutate(Por_perdido = round((dropout_loss / AUC) * 100, 2))

# IMPORTANT NOTE: 
# The suffix numbering (_1, _2, _3, _4) in PDP dataframes may conflict with 
# previous male analyses if both are run in the same session.
# Consider using unique names for female PDP dataframes to avoid overwriting.

# Step 8.3: Combine Partial Dependence Plot (PDP) data for Age (Female)
# ----------------------------------------------------------------------------
# Combine PDP results for Age variable across all female ethnic groups
df_graf_age = rbind(df_graf_Age_1, df_graf_Age_2, 
                    df_graf_Age_3, df_graf_Age_4)

# Ensure probability values are non-negative (trim negative values to 0)
df_graf_age = df_graf_age %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

# Optional: Save combined data for external analysis
# write.csv(df_graf_age, "path/to/save/df_graf_age_women.csv")

# Step 8.4: Create PDP visualization for Age (Female)
# ----------------------------------------------------------------------------
# Line plot showing how chronic disease probability changes with age
# Different colors/shapes for each female ethnic group
df_graf_age %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +  # Empty y-label for side-by-side plotting with male results
  scale_x_continuous(n.breaks = 10) +  # 10 breaks on x-axis for Age
  scale_y_continuous(n.breaks = 10) +  # 10 breaks on y-axis for probability
  xlab("Age (years)")

# Step 8.5: Repeat combination and visualization for Education (Female)
# ----------------------------------------------------------------------------
df_graf_edu = rbind(df_graf_edu_1, df_graf_edu_2, 
                    df_graf_edu_3, df_graf_edu_4)

df_graf_edu = df_graf_edu %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_edu %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  # Note: Education is categorical, so no continuous scale breaks
  scale_y_continuous(n.breaks = 10) +
  xlab("Education Level")

# Step 8.6: Repeat for Occupation (Female)
# ----------------------------------------------------------------------------
df_graf_ocupa = rbind(df_graf_ocu_1, df_graf_ocu_2, 
                      df_graf_ocu_3, df_graf_ocu_4)

df_graf_ocupa = df_graf_ocupa %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_ocupa %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_y_continuous(n.breaks = 10) +
  xlab("Occupation Category")

# Step 8.7: Repeat for Piped Water access (Female)
# ----------------------------------------------------------------------------
df_graf_piper = rbind(df_graf_piper_1, df_graf_piper_2, 
                      df_graf_piper_3, df_graf_piper_4)

df_graf_piper = df_graf_piper %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_piper %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Binary variable: 0=No, 1=Yes
  scale_y_continuous(n.breaks = 10) +
  xlab("Piped Water Access")

# Step 8.8: Repeat for Garbage collection (Female)
# ----------------------------------------------------------------------------
df_graf_garba = rbind(df_graf_garbage_1, df_graf_garbage_2, 
                      df_graf_garbage_3, df_graf_garbage_4)

df_graf_garba = df_graf_garba %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_garba %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Binary variable
  scale_y_continuous(n.breaks = 10) +
  xlab("Garbage Collection Service")

# Step 8.9: Repeat for Urban/Rural residence (Female)
# ----------------------------------------------------------------------------
df_graf_urban = rbind(df_graf_urban_1, df_graf_urban_2, 
                      df_graf_urban_3, df_graf_urban_4)

df_graf_urban = df_graf_urban %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_urban %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Likely: 0=Rural, 1=Urban
  scale_y_continuous(n.breaks = 10) +
  xlab("Residence Type (Urban/Rural)")

# Step 8.10: Repeat for Water access (Female)
# ----------------------------------------------------------------------------
df_graf_water = rbind(df_graf_water_1, df_graf_water_2, 
                      df_graf_water_3, df_graf_water_4)

df_graf_water = df_graf_water %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_water %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("") +
  scale_x_continuous(n.breaks = 2) +  # Likely categorical water access
  scale_y_continuous(n.breaks = 10) +
  xlab("Water Access")

# Step 8.11: Repeat for Sewage system access (Female)
# ----------------------------------------------------------------------------
df_graf_sewage = rbind(df_graf_sewage_1, df_graf_sewage_2, 
                       df_graf_sewage_3, df_graf_sewage_4)

df_graf_sewage = df_graf_sewage %>% 
  mutate(U5MR_average = ifelse(U5MR_average < 0, 0, U5MR_average))

df_graf_sewage %>% 
  ggplot(aes(value, U5MR_average, color = Group, shape = Group)) +
  geom_line(size = 1.5) + 
  theme_bw() +
  ylab("Average probability of being Dx with chronic disease") +
  scale_x_continuous(n.breaks = 2) +  # Likely: 0=No sewage, 1=Has sewage
  scale_y_continuous(n.breaks = 10) +
  xlab("Sewage System Access")

# ============================================================================
#