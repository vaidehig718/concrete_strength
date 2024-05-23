import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# Create an instance of XGBoost regressor
model = xgb.XGBRegressor()  # For regression tasks

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict using the trained model on the validation set
y_pred = model.predict(X_val)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_val, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_val, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(y_val, y_pred)

# Calculate Explained Variance Score
explained_variance = explained_variance_score(y_val, y_pred)

# Print the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)
print("Explained Variance Score:", explained_variance)


from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their respective values to search
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'max_depth': [3, 5, 7],
    'n_estimators': [100, 200, 300],
}

# Create an instance of XGBoost regressor
model = xgb.XGBRegressor()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Retrain the model using the best hyperparameters
best_model.fit(X_train, y_train)

# Evaluate the best model on the validation set
y_pred_best = best_model.predict(X_val)

# Calculate the evaluation metrics for the best model
mse_best = mean_squared_error(y_val, y_pred_best)
mae_best = mean_absolute_error(y_val, y_pred_best)
r2_best = r2_score(y_val, y_pred_best)
explained_variance_best = explained_variance_score(y_val, y_pred_best)

# Print the evaluation metrics for the best model
print("Best Model - Mean Squared Error (MSE):", mse_best)
print("Best Model - Mean Absolute Error (MAE):", mae_best)
print("Best Model - R-squared (R2) Score:", r2_best)
print("Best Model - Explained Variance Score:", explained_variance_best)



import numpy as np

# Combine the training and validation data
X_train_combined = np.concatenate((X_train, X_val))
y_train_combined = np.concatenate((y_train, y_val))

# Create an instance of XGBoost regressor with the best hyperparameters
model_retrained = xgb.XGBRegressor(**best_params)

# Retrain the model using the combined training and validation data
model_retrained.fit(X_train_combined, y_train_combined)

# Predict using the retrained model on the test set
y_pred_test = model_retrained.predict(X_test)

# Calculate the evaluation metrics on the test set
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
explained_variance_test = explained_variance_score(y_test, y_pred_test)

# Print the evaluation metrics on the test set
print("Retrained Model - Mean Squared Error (MSE):", mse_test)
print("Retrained Model - Mean Absolute Error (MAE):", mae_test)
print("Retrained Model - R-squared (R2) Score:", r2_test)
print("Retrained Model - Explained Variance Score:", explained_variance_test)





