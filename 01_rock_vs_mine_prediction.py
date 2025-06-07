# Rock vs Mine Prediction using Sonar Dataset
# Import necessary libraries
import pandas as pd # Data manipulation and analysis
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization
from sklearn.model_selection import train_test_split # Train-test split
from sklearn.linear_model import LogisticRegression # Logistic Regression model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Model evaluation metrics

# Load the dataset
df = pd.read_csv('datasets/sonar_data.csv', header=None)  # Header=None indicates no header row in the dataset

# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Display the shape of the dataset
df.shape

# Display the class distribution
df[60].value_counts()

# Describe the dataset
df.describe().T

# Visualize the distribution of classes
sns.countplot(x=df[60], data=df)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count') 
plt.show()

# Getting mean of each feature grouped by class
df.groupby(60).mean()

# Split the dataset into features and target variable
X = df.drop(60, axis=1)
y = df[60]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=1)
X.shape, X_train.shape, X_test.shape

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy score of the model on the training set
X_train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, X_train_predictions)
print(f'Training Accuracy: {train_accuracy:.2f}')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix') 
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Visualize the coefficients of the model
coefficients = pd.DataFrame(model.coef_[0], index=X.columns, columns=['Coefficient'])
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients['Coefficient'], y=coefficients.index)
plt.title('Feature Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()

# Make predictions on a new sample
new_sample = X_train.sample(n=1, random_state=42)
# Predict the class of the new sample
new_prediction = model.predict(new_sample)

if new_prediction[0] == 'M': print("The new sample is predicted to be a Mine.")
else: print("The new sample is predicted to be a Rock.")

# Save the model for future use
import joblib
joblib.dump(model, 'rock_vs_mine_model.pkl')
# Load the model to verify it works
loaded_model = joblib.load('rock_vs_mine_model.pkl')
# Verify loaded model predictions
loaded_predictions = loaded_model.predict(X_test)
# Check if the loaded model predictions match the original predictions
assert np.array_equal(y_pred, loaded_predictions), "Loaded model predictions do not match original predictions."
# End of the rock vs mine prediction project
# This code completes the rock vs mine prediction project using the Sonar dataset.
# The model has been trained, evaluated, and saved successfully.