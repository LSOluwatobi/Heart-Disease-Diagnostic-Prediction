# ==============================================================================
# PROJECT: Heart Disease Prediction
# ML Task: Classification
# Models: Logistic Regression, LDA, Naive Bayes
# ==============================================================================

# ------------------------------------------------------------------------------
# PHASE 1: Setup & Libraries
# ------------------------------------------------------------------------------
if(!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, caret, MASS, e1071, pROC, corrplot, car, MVN, rstatix)

# ------------------------------------------------------------------------------
# PHASE 2: Data Loading & Exploration
# ------------------------------------------------------------------------------
# Loading the Processed Cleveland Dataset
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names <- c("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target")

# Corrected Loading Line
heart_data <- read.csv(url, header = FALSE, col.names = names, na.strings = "?")

# Initial Exploration
summary(heart_data)
str(heart_data)
head(heart_data)

# Visualizing Target Distribution
# Target: 0 = No Disease, 1-4 = Presence of Disease
table(heart_data$target)

# ------------------------------------------------------------------------------
# PHASE 3: Data Cleaning
# ------------------------------------------------------------------------------
# 1. Handle Missing Values (NAs exist in 'ca' and 'thal')
heart_data <- heart_data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# 2. Binary Target Transformation
# Converting multiclass target (0,1,2,3,4) to binary (0 = No, 1 = Yes)
heart_data$target <- ifelse(heart_data$target > 0, 1, 0)
heart_data$target <- factor(heart_data$target, levels = c(0, 1), labels = c("No", "Yes"))

# 3. Categorical Factorization
categorical_vars <- c("sex", "cp", "fbs", "restecg", "exang", "slope")
heart_data[categorical_vars] <- lapply(heart_data[categorical_vars], as.factor)

# ------------------------------------------------------------------------------
# PHASE 4: Feature Engineering (Encoding & Scaling)
# ------------------------------------------------------------------------------
# One-Hot Encoding for multi-level categorical variables (like chest pain type)
dummies <- dummyVars(~ ., data = heart_data[, -which(names(heart_data) == "target")], fullRank = TRUE)
features_dummy <- data.frame(predict(dummies, newdata = heart_data))

# Standardization (Crucial for Parametric Models)
scaler <- preProcess(features_dummy, method = c("center", "scale"))
scaled_features <- predict(scaler, features_dummy)

final_df <- cbind(target = heart_data$target, scaled_features)

# ------------------------------------------------------------------------------
# PHASE 5: Assumption Diagnostics
# ------------------------------------------------------------------------------
# 1. Multicollinearity (VIF)
vif_mod <- glm(target ~ ., data = final_df, family = binomial)
print(vif(vif_mod))

# 2. Correlation Plot
cor_mat <- cor(scaled_features)
corrplot(cor_mat, method = "color", type = "upper", tl.cex = 0.7, 
         title = "Heart Disease Feature Correlations", mar=c(0,0,1,0))

# ------------------------------------------------------------------------------
# PHASE 6: Data Partitioning
# ------------------------------------------------------------------------------
trainIndex <- createDataPartition(final_df$target, p = 0.8, list = FALSE)
train_set <- final_df[trainIndex, ]
test_set  <- final_df[-trainIndex, ]

# ------------------------------------------------------------------------------
# PHASE 7: Model Training
# ------------------------------------------------------------------------------
# Logistic Regression
fit_logit <- glm(target ~ ., data = train_set, family = binomial)

# LDA
fit_lda <- lda(target ~ ., data = train_set)

# Naive Bayes
fit_nb <- naiveBayes(target ~ ., data = train_set)

# ------------------------------------------------------------------------------
# PHASE 8: Evaluation & Performance Metrics
# ------------------------------------------------------------------------------
# Predicted Probabilities
p_logit <- predict(fit_logit, test_set, type = "response")
p_lda   <- predict(fit_lda, test_set)$posterior[, "Yes"]
p_nb    <- predict(fit_nb, test_set, type = "raw")[, "Yes"]

get_metrics <- function(probs, actual, label) {
  classes <- factor(ifelse(probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
  cm <- confusionMatrix(classes, actual, positive = "Yes", mode = "everything")
  roc_obj <- roc(actual, probs, quiet = TRUE)
  
  return(c(Model = label,
           Accuracy = cm$overall["Accuracy"],
           ROC_AUC = auc(roc_obj),
           Precision = cm$byClass["Precision"],
           Recall = cm$byClass["Recall"],
           F1_Score = cm$byClass["F1"]))
}

comparison_df <- rbind(
  get_metrics(p_logit, test_set$target, "Logistic Regression"),
  get_metrics(p_lda, test_set$target, "LDA"),
  get_metrics(p_nb, test_set$target, "Naive Bayes")
) %>% as.data.frame()

print(comparison_df)

# ------------------------------------------------------------------------------
# PHASE 9: Visual Comparison (ROC Plot)
# ------------------------------------------------------------------------------
plot(roc(test_set$target, p_logit), col="blue", lwd=2, main="ROC Curve: Heart Disease Prediction")
plot(roc(test_set$target, p_lda), col="red", lwd=2, add=TRUE)
plot(roc(test_set$target, p_nb), col="green", lwd=2, add=TRUE)
legend("bottomright", legend=c("Logit", "LDA", "Naive Bayes"), col=c("blue", "red", "green"), lwd=2)
