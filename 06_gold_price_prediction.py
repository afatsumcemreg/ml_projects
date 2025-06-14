# Gold price prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Import necessary libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, RobustScaler

# Load the dataset 
df = pd.read_csv(r'C:\Users\musta\hands_on_machine_learning\datasets\gold_price_data.csv')

# Write a function to check the dataframe
def check_dataframe(df):
    print("DataFrame Shape:", df.shape)
    print("DataFrame Columns:", df.columns)
    print("DataFrame Info:")
    print(df.info())
    print("DataFrame Head:")
    print(df.head())
    print("DataFrame Tail:")
    print(df.tail())
    print("DataFrame Description:")
    print(df.describe())
    print("Missing Values:")
    print(df.isnull().sum())

# Check the dataframe
check_dataframe(df)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Visualize the GLD value over time 
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['GLD'], label='GLD Value', color='blue')
plt.title('GLD Value Over Time')
plt.xlabel('Date')
plt.ylabel('GLD Value')
plt.legend()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8)) 
sns.heatmap(df.drop('Date', axis=1).corr(), annot=True, fmt='.2f', cmap='Blues', square=True)
plt.title('Correlation Matrix')
plt.show()

# Correlation with GLD
df.drop('Date', axis=1).corr()['GLD']

# Distribution of GLD values
plt.figure(figsize=(10, 6))
sns.histplot(df['GLD'], bins=30, kde=True, color='blue')
plt.title('Distribution of GLD Values')
plt.xlabel('GLD Value')
plt.ylabel('Frequency') 
plt.show()

# Time series decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.set_index('Date')['GLD'], model='additive', period=30)
result.plot()
plt.show()

# Feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Weekday'] = df['Date'].dt.weekday
df['Quarter'] = df['Date'].dt.quarter

# Drop the original 'Date' column
df.drop('Date', axis=1, inplace=True)
# Check the updated dataframe
check_dataframe(df)
# Split the dataset into features and target variable
X = df.drop('GLD', axis=1)
y = df['GLD']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# Fit the model on the training data
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the training and testing sets
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'R^2 Score: {r2:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')

# Evaluate the model on training and testing sets
print("Training Set Evaluation:")
evaluate_model(y_train, y_train_pred)
print("\nTesting Set Evaluation:")
evaluate_model(y_test, y_test_pred)

# Feature importance
feature_importances = rf_model.feature_importances_
feature_names = X.columns
# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Save the model
import joblib
joblib.dump(rf_model, 'gold_price_prediction_model.pkl')
# Load the model
loaded_model = joblib.load('gold_price_prediction_model.pkl')
# Make predictions with the loaded model    
loaded_y_test_pred = loaded_model.predict(X_test_scaled)
# Evaluate the loaded model   
print("\nLoaded Model Evaluation:")
evaluate_model(y_test, loaded_y_test_pred)

# Visualize the predictions vs actual values using scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, loaded_y_test_pred, alpha=0.5)
plt.title('Actual vs Predicted GLD Values')
plt.xlabel('Actual GLD Value')
plt.ylabel('Predicted GLD Value')
plt.plot([y_test.min(), y_test.max()], [loaded_y_test_pred.min(), loaded_y_test_pred.max()], color='red', linestyle='--')
plt.xlim(y_test.min(), y_test.max())
plt.ylim(loaded_y_test_pred.min(), loaded_y_test_pred.max())
plt.show()

y_test_pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': loaded_y_test_pred})
# Compare actual vs predicted values in a plot
y_test = list(y_test)
plt.figure(figsize=(14, 7))
plt.plot(y_test, label='Actual GLD Value', color='blue')
plt.plot(loaded_y_test_pred, label='Predicted GLD Value', color='green', linestyle='--')     
plt.title('Actual vs Predicted GLD Values')
plt.xlabel('Number of Observations')
plt.ylabel('GLD Value')
plt.legend()
plt.show()

# Predicting GLD value for new data
new_data_train = X_train.sample(1, random_state=42)
new_data_train_scaled = scaler.transform(new_data_train)
predicted_gld_value_train = loaded_model.predict(new_data_train_scaled)
print(f'Predicted GLD Value for New Data: {predicted_gld_value_train[0]:.2f}')

new_data_test = X_test.sample(1, random_state=42)
new_data_test_scaled = scaler.transform(new_data_test)
predicted_gld_value_test = loaded_model.predict(new_data_test_scaled)
print(f'Predicted GLD Value for New Data: {predicted_gld_value_test[0]:.2f}')