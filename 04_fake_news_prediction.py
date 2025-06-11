# Fake News Prediction using machine learning
# Import necessary libraries
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set display options for pandas
pd.set_option('display.max_columns', None)  # To display all columns in DataFrame
pd.set_option('display.width', 500)  # To display all rows in DataFrame

# Load the dataset
df = pd.read_csv('datasets/FakeNewsNet.csv')

# Download NLTK resources
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Display the shape of the dataset
df.shape
# Display the first few rows of the dataset
df.head()

# Missing value analysis
df.isnull().sum()

# Define a function to check dataframe
def check_dataframe(df):
    print("DataFrame Shape:", df.shape)
    print("DataFrame Columns:", df.columns.tolist())
    print("DataFrame Info:")
    print(df.info())
    print("DataFrame Head:")
    print(df.head())
    print("DataFrame Description:")
    print(df.describe())
    print("Missing Values:")
    print(df.isnull().sum())
    print("DataFrame Sample:")
    print(df.sample(5))
    print("DataFrame Dtypes:")
    print(df.dtypes)
    print("DataFrame Memory Usage:")
    print(df.memory_usage(deep=True))

# Check the dataframe
check_dataframe(df)

# Replacing the null values with an empty string
df = df.fillna('')

# Missing value analysis after filling nulls
df.isnull().sum()

for col in df.columns:
    print(df[col].head(10))

# Getting feature and target variables
X = df.drop('real', axis=1)  # Features
y = df['real'] # Target variable

# Stemming 
# Initialize the Porter Stemmer
port_stemm = PorterStemmer()
# Function to perform stemming
def stemming(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters	
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+[a-z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'^[a-z]\s+', ' ', text)  # Remove single characters at the start
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.strip()  # Remove leading and trailing spaces
    text = ' '.join([port_stemm.stem(word) for word in text.split() if word not in stop_words])
    return text

df['title'] = df['title'].apply(stemming)

# Separating the data and target
X = df['title'].values
y = df['real'].values
X.shape, y.shape

# Convertion of text to feature vectors
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)  # Fit and transform the text data
print(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Display the shapes of the training and testing sets
X.shape, X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Model training using Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Model prediction using tranign data
y_train_pred = model.predict(X_train)
# Model prediction using testing data
y_test_pred = model.predict(X_test)
# Display the accuracy of the model on training and testing data
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

# Display the classification report for training and testing data
print("Training Classification Report:\n", classification_report(y_train, y_train_pred))
print("Testing Classification Report:\n", classification_report(y_test, y_test_pred))
# Display the confusion matrix for training and testing data    
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print("Testing Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Function to predict fake news
def predict_fake_news(text):
    text = stemming(text)  # Preprocess the text
    text_vectorized = tfidf.transform([text])  # Convert text to feature vector
    prediction = model.predict(text_vectorized)  # Predict using the model
    return "Fake News" if prediction[0] == 0 else "Real News"

# Example usage of the prediction function
print(predict_fake_news("The stock market is crashing!"))
print(predict_fake_news("The president signed a new bill into law."))
# Example usage of the prediction function with a custom text
custom_text = "Scientists discover a new species of bird in the Amazon rainforest."
print(predict_fake_news(custom_text))
# Example usage of the prediction function with another custom text
custom_text2 = "The new smartphone model has a revolutionary camera feature."
print(predict_fake_news(custom_text2))

x_new = X_test[6]
prediction = model.predict(x_new)
print("Predicted class for the new sample:", prediction[0])

if prediction[0] == 0:
    print("The news is fake.")  
else:
    print("The news is real.")