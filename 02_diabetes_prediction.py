# Diabetes prediction using support vector machines (SVM)
# This script uses a dataset to predict diabetes using SVM.
# Libraries used: pandas, numpy, matplotlib, seaborn, scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)  # Show all columns in the DataFrame
pd.set_option('display.width', 1000)  # Set the display width for better readability
pd.set_option('display.float_format', '{:.2f}'.format)  # Set float format for better readability

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
diabetes_dataset = pd.read_csv('datasets/diabetes.csv')
df = diabetes_dataset.copy()

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
df.isnull().sum()

# Check the distribution of the target variable
print(df['Outcome'].value_counts())

# Visualize the distribution of the target variable
sns.countplot(x='Outcome', data=df)
plt.title('Distribution of Diabetes Outcome')
plt.show()

# Describe the dataset 
df.describe().T

# Shape of the dataset
df.shape

# Group by 'Outcome' and calculate the mean of each feature
df.groupby('Outcome').mean()

# Split the dataset into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X.shape, y.shape

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
# Stratify the split to maintain the proportion of classes in both sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=2)
X.shape, X_train.shape, X_test.shape

# Create a pipeline with SVM classifier 
pipeline = Pipeline([
    ('svm', SVC(kernel='linear', random_state=42))
])

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = pipeline.predict(X_train)
# Evaluate the model on the training data
print("Training Data Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))
# Calculate training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")

# Make predictions on the test data
y_pred = pipeline.predict(X_test)   
# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Calculate test accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize the feature importance using a bar plot 
feature_importance = np.abs(pipeline.named_steps['svm'].coef_[0])
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.show()

# Making predictions on new data
# Reshape the new data to match the input shape of the model
new_data = X_train[5].reshape(1, -1)  # Sample 5 rows from the training data
new_predictions = pipeline.predict(new_data)  # Predict using the trained model

new_data = X.sample(n=1, random_state=42)  # Sample a new data point
new_data_scaled = scaler.transform(new_data)  # Scale the new data
new_prediction = pipeline.predict(new_data_scaled)  # Predict using the trained model
print(f"New Data Sample:\n{new_data}\nPrediction: {new_prediction[0]}")  # Display the new data and its prediction

# Interpret and print the prediction result for the new data sample
if new_prediction[0] == 1: print("The model predicts that the individual has diabetes.")
else: print("The model predicts that the individual does not have diabetes.")

# Save the model for future use
import joblib
joblib.dump(pipeline, 'diabetes_svm_model.pkl') 
# Load the model to verify it works
loaded_model = joblib.load('diabetes_svm_model.pkl')
# Make predictions with the loaded model
loaded_y_pred = loaded_model.predict(X_test)
# Verify the loaded model's predictions match the original model's predictions
assert np.array_equal(y_pred, loaded_y_pred), "Loaded model predictions do not match original model predictions."   
print("Model saved and loaded successfully. Predictions match.")
# End of the diabetes prediction script
# This script demonstrates how to use SVM for diabetes prediction using a dataset.
# The model is saved using joblib for future use.
# The script includes data preprocessing, model training, evaluation, and visualization.
# The model can be used to predict diabetes outcomes based on the features in the dataset.
# The script is complete and ready for use.
# End of the diabetes prediction script