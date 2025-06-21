# Wine quality prediction
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils.helpers_class import preprocessing
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%-3f' % x)
# Set seaborn figure style for ggplot-like aesthetics
sns.set_style("ticks")

# Load the dataset from the folder called 'datesets'
df = pd.read_csv('datasets/winequality-red.csv')

# Call the preprocessing class
prep = preprocessing(df)
prep.check_df()
cat_cols, num_cols, cat_but_car = prep.grab_col_names()
prep.cat_summary(cat_cols, True)
prep.num_summary(num_cols, plot=True)
prep.target_summary_with_num('quality', num_cols, plot=True)
prep.high_correlated_cols(num_cols, plot=True)

sns.catplot(x='quality', data=df, kind='count', palette='Set2')
plt.show()

# Split the dataset into features and target variable
X = df.drop('quality', axis=1)

# Label binarize the target variable
y = df['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Model training
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Model prediction using the training set
y_train_pred = rf_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Model prediction using the testing set
y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the accuracy scores
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Classification report for the testing set
print("\nClassification Report (Testing Set):")
print(classification_report(y_test, y_test_pred))
# Confusion matrix for the testing set
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
              xticklabels=['Low Quality', 'High Quality'],
              yticklabels=['Low Quality', 'High Quality'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature importance
feature_importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Save the model using joblib
import joblib       
joblib.dump(rf_model, 'wine_quality_rf_model.pkl')

# Load the model to verify it works
loaded_model = joblib.load('wine_quality_rf_model.pkl')
loaded_model_pred = loaded_model.predict(X_test)    
# Check if the loaded model gives the same predictions
assert np.array_equal(y_test_pred, loaded_model_pred), "Loaded model predictions do not match original predictions."
print("Model loaded successfully and predictions match.")

# Building a predictitive system
sample_data = X_train.sample(1, random_state=42)
sample_prediction = rf_model.predict(sample_data)
print(f"Sample Data:\n{sample_data}")
print(f"Predicted Quality for Sample Data: {'High Quality' if sample_prediction[0] == 1 else 'Low Quality'}")

sample_data_test = X_test.sample(1, random_state=42)
sample_prediction_test = rf_model.predict(sample_data_test)
print(f"Sample Data Test:\n{sample_data_test}")
print(f"Predicted Quality for Sample Data Test: {'High Quality' if sample_prediction_test[0] == 1 else 'Low Quality'}")

input_data = [7.2,0.36,0.46,2.1,0.07400000000000001,24.0,44.0,0.99534,3.4,0.85,11.0]
input_data = np.asarray(input_data).reshape(1, -1)
input_prediction = rf_model.predict(input_data)
print(f"Input Data: {input_data}")  
print(f"Predicted Quality for Input Data: {'High Quality' if input_prediction[0] == 1 else 'Low Quality'}")