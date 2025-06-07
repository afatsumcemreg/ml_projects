# House price prediction with XGBoost Regressor
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

# California dataset
from sklearn.datasets import fetch_california_housing

# Set display options for better readability
pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
pd.set_option('display.width', 1000)  # Set the display width for better readability
pd.set_option('display.float_format', '{:.2f}'.format)  # Set float format for better readability
# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
house_price_dataset = fetch_california_housing(as_frame=True)
df = house_price_dataset.data
df['price'] = house_price_dataset.target  # Add target variable to the DataFrame
# Display the first few rows of the dataset
df.head()
# Check for missing values
df.isnull().sum()
# Shape of dataset
df.shape
# Describe the dataset
df.describe().T
# Visualize the distribution of the target variable
sns.histplot(df['price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')
plt.show()
# Check the correlation between features and target variable
correlation = df.corr()['price'].sort_values(ascending=False)
print(correlation)
# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
# Split the dataset into features and target variable
X = df.drop('price', axis=1)
y = df['price']
X.shape, y.shape
# Standardize the features using RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2)
X.shape, X_train.shape, X_test.shape
# Create and train the XGBoost Regressor model
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=2)
model.fit(X_train, y_train)
# Make predictions on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Evaluate the model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
# Print the evaluation metrics
print(f'Training MSE: {train_mse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.2f}')
print(f'Testing MSE: {test_mse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.2f}')
# Visualize the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Predicted vs Actual House Prices')
plt.xlabel('Actual House Price')
plt.ylabel('Predicted House Price')
plt.xlim(y.min(), y.max())
plt.ylim(y.min(), y.max())
plt.grid()
plt.show()
# Visualize feature importance
plt.figure(figsize=(12, 8))
plt.barh(X.columns, model.feature_importances_, color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()
# Save the model for future use
import joblib
joblib.dump(model, 'xgboost_house_price_model.pkl')
# Load the model
loaded_model = joblib.load('xgboost_house_price_model.pkl')
# Make predictions with the loaded model
loaded_y_test_pred = loaded_model.predict(X_test)
# Verify that the predictions are the same
assert np.allclose(y_test_pred, loaded_y_test_pred), "Loaded model predictions do not match original predictions."
# The model has been successfully trained, evaluated, and saved for future use.
# The code above demonstrates how to perform house price prediction using the XGBoost Regressor on the California housing dataset.
# Making predictions on new data
# Sample a new data point from the training set
new_data = X_train[5].reshape(1, -1)  # Reshape to match the input shape of the model
# Scale the new data using the same scaler
new_data_scaled = scaler.transform(new_data)
# Predict the house price using the trained model
new_prediction = model.predict(new_data_scaled)
# Display the new data and its prediction
print(f"New Data Sample:\n{new_data}\nPredicted House Price: {new_prediction[0]:.2f}")