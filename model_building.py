import statsmodels.api as sm

# Preparing the data
X = df[['Superplasticizer', 'Age_Days', 'Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Coarse_Aggregate', 'Fine_Aggregate']]  # Predictor variables
y = df['Strength']  # Target variable

# Adding a constant term to the predictor variables
X = sm.add_constant(X)

# Fitting the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Printing OLS regression results
print(results.summary())

# Fit the linear regression model
model = sm.OLS(y, X)
results = model.fit()

# Calculate the residuals
residuals = results.resid

# Plot the residuals
import matplotlib.pyplot as plt

plt.scatter(y, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Target Variable')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# Plot scatter plots of each predictor variable against the target variable
import seaborn as sns

sns.pairplot(df, x_vars=['Superplasticizer', 'Age_Days', 'Cement', 'Blast_Furnace_Slag', 'Water', 'Coarse_Aggregate', 'Fine_Aggregate'], y_vars='Strength', kind='scatter')
plt.show()


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Fit a linear regression model
lr_model = LinearRegression()
lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
linear_score = lr_scores.mean()

# Fit a nonlinear regression model (Decision Tree)
dt_model = DecisionTreeRegressor()
dt_scores = cross_val_score(dt_model, X, y, cv=5, scoring='r2')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost regression model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Calculate R-squared for XGBoost
r2_xgb = r2_score(y_test, y_pred_xgb)

# Store the results in a dictionary
results = {
    "Linear Regression": linear_score,
    "Decision Tree Regression (Cross-Validation)": dt_scores.mean(),
    "XGBoost Regression": r2_xgb,
}

# Sort the results by R-squared score in descending order
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

# Get the best model
best_model = sorted_results[0][0]

# Print the final summary
print("Model Comparison:")
for model, score in sorted_results:
    print(f"{model}: {score}")
print("\nBest Model:", best_model)

# splitting the datasest into 3 datasets: train, validate, and test
train, test = train_test_split(df, test_size=.2, random_state=42)
train, val = train_test_split(train, test_size=.1, random_state=42)

# Splitting the dataset into features (X) and target variable (y)
X_train = train.drop('Strength', axis=1)
y_train = train['Strength']

X_val = val.drop('Strength', axis=1)
y_val = val['Strength']

X_test = test.drop('Strength', axis=1)
y_test = test['Strength']






