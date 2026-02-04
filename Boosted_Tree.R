# Load xgboost library
library(xgboost)

label_for_boost <- as.numeric(as.character(y))

set.seed(82)
total_rows <- length(label_for_boost)
rows_for_validation <- sample(1:total_rows, size = floor(0.2 * total_rows))
rows_for_training <- setdiff(1:total_rows, rows_for_validation)

X_train_boost <- X[rows_for_training, ]
y_train_boost <- label_for_boost[rows_for_training]
X_vali_boost <- X[rows_for_validation, ]
y_vali_boost <- label_for_boost[rows_for_validation]

dtrain_boosted <- xgb.DMatrix(data = as.matrix(X_train_boost), label = y_train_boost)
dvali_boosted <- xgb.DMatrix(data = as.matrix(X_vali_boost), label = y_vali_boost)
dtest_boosted <- xgb.DMatrix(data = as.matrix(test.X), label = as.numeric(as.character(test.y)))

tree_amounts <- c(10000, 20000, 30000)
depth_amounts <- c(4, 5, 6)
learning_value <- 0.001

lowest_wrong <- 1
best_combo <- c()

for (trees_try in tree_amounts) {
  for (depth_try in depth_amounts) {
    
    
    boosting_model_try <- xgboost(data = dtrain_boosted,
                                  max.depth = depth_try,
                                  eta = learning_value,
                                  nrounds = trees_try,
                                  objective = "binary:logistic",
    )
    
    pred_vali_try <- predict(boosting_model_try, newdata = dvali_boosted)
    pred_vali_try_binary <- ifelse(pred_vali_try > 0.5, 1, 0)
    error_vali_try <- mean(pred_vali_try_binary != y_vali_boost)
    
    cat("Trees:", trees_try, "| Depth:", depth_try,
        "| Validation Error Rate:", round(error_vali_try, 4), "\n")
    
    if (error_vali_try < lowest_wrong) {
      lowest_wrong <- error_vali_try
      best_combo <- c(trees_try, depth_try)
    }
  }
}

cat("\nBest Parameters → Trees:", best_combo[1],
    "| Depth:", best_combo[2], "| Validation Error:", round(lowest_wrong, 4), "\n")

final_training_boost <- xgb.DMatrix(data = as.matrix(X), label = label_for_boost)
final_model_boosting <- xgboost(
  data = final_training_boost,
  max.depth = best_combo[2],
  eta = 0.001,              # required by assignment
  nrounds = best_combo[1],
  objective = "binary:logistic"
)

prediction_test_boost <- predict(final_model_boosting, newdata = dtest_boosted)
prediction_test_boost_binary <- ifelse(prediction_test_boost > 0.5, 1, 0)
final_error_test_boost <- mean(prediction_test_boost_binary != as.numeric(as.character(test.y)))

cat("\nTest Misclassification Rate (Boosted Tree):", round(final_error_test_boost, 4), "\n")

###################### result ###############################
#Trees: 10000 | Depth: 4 | Validation Error Rate: 0.0129 
#Trees: 10000 | Depth: 5 | Validation Error Rate: 0.0109 
#Trees: 10000 | Depth: 6 | Validation Error Rate: 0.0121 
#Trees: 20000 | Depth: 4 | Validation Error Rate: 0.01 
#Trees: 20000 | Depth: 5 | Validation Error Rate: 0.0083 
#Trees: 20000 | Depth: 6 | Validation Error Rate: 0.0104 
#Trees: 30000 | Depth: 4 | Validation Error Rate: 0.0092 
#Trees: 30000 | Depth: 5 | Validation Error Rate: 0.0083 
#Trees: 30000 | Depth: 6 | Validation Error Rate: 0.0092 
#Best Parameters → Trees: 20000 | Depth: 5 | Validation Error: 0.0083 
#Test Misclassification Rate (Boosted Tree): 0.0055 
#######################################################################

