# Loan status prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the style for seaborn
sns.set(style='whitegrid')

# Set display options for pandas
pd.set_option('display.width', 300)
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv('datasets/loan_data.csv')

# Define a function to check the dataframe
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

# Data Preprocessing
# Drop missing values
df.dropna(inplace=True)

# LABEL ENCODING: METHOD 1
# Label encoding for the categorical variables using replace method
def label_encode_column(df, column):
    unique_values = df[column].unique()
    mapping = {value: idx for idx, value in enumerate(unique_values)}
    df[column] = df[column].replace(mapping)
    return df

# Apply label encoding to categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('Loan_ID')  # Exclude 'Loan_ID' from encoding
for column in categorical_columns:
    df = label_encode_column(df, column)
# Check the dataframe after preprocessing
check_dataframe(df)

# LABEL ENCODING: METHOD 2
# Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
categorical_columns = categorical_columns.drop('Loan_ID')
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)
# Check the dataframe after preprocessing
check_dataframe(df)

for column in categorical_columns:
    print(df[column].value_counts())

# Data visualization of categorical variables
def plot_categorical_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df, hue='Loan_Status', palette='Set2')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45)
    plt.show()

# Plot categorical distributions
for column in categorical_columns:
    plot_categorical_distribution(df, column)

# Data visualization of numerical variables
def plot_numerical_distribution(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, color='blue')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.show()

# Plot numerical distributions
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    plot_numerical_distribution(df, column)

# Separate features and target variable
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Train a Support Vector Machine (SVM) model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

# Print confusion matrix and classification report
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_test_pred))
print("Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Visualize the confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Plot confusion matrix for the test set
cm = confusion_matrix(y_test, y_test_pred)
plot_confusion_matrix(cm, classes=['Not Approved', 'Approved'])

# Save the model for future use
import joblib
joblib.dump(model, 'loan_status_model.pkl')

# Load the model
model = joblib.load('loan_status_model.pkl')

# Example usage of the model
new_data = X_train.sample(n=1, random_state=42)
new_prediction = model.predict(new_data)
print("New Data Sample:")
print(new_data)
print("Prediction for New Data Sample:", new_prediction)

# Example usage of the model
new_data = X_test.sample(n=1, random_state=42)
new_prediction = model.predict(new_data)
print("New Data Sample:")
print(new_data)
print('Prediction for New Data Sample:', new_prediction)