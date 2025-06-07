# MODEL-BASED MACHINE LEARNING WITH LINEAR REGRESSION
# This script demonstrates how to load a dataset, visualize it, and fit a linear regression model using Python.
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset  
data_link = 'https://github.com/ageron/data/raw/main/'
df = pd.read_csv(data_link + 'lifesat/lifesat.csv')

# Display the first few rows of the dataset
df.head()


X = df[['GDP per capita (USD)']].values # Reshape to 2D array
y = df['Life satisfaction'].values # Reshape to 1D array  
  
# Create a scatter plot
df.plot(kind='scatter', x='GDP per capita (USD)', y='Life satisfaction', alpha=0.5, grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.title('Life Satisfaction vs GDP per Capita')
plt.xlabel('GDP per Capita (USD)')
plt.ylabel('Life Satisfaction')
plt.show()

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict life satisfaction for a new GDP per capita value
X_new = np.array([[37_655.2]])
y_pred = model.predict(X_new)
print(f"Predicted life satisfaction for GDP per capita of {X_new[0][0]}: {y_pred[0]:.2f}")

# INSTANCE-BASED MACHINE LEARNING WITH KNN
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

data_link = 'https://github.com/ageron/data/raw/main/'
df = pd.read_csv(data_link + 'lifesat/lifesat.csv')

# Display the first few rows of the dataset
df.head()


X = df[['GDP per capita (USD)']].values # Reshape to 2D array
y = df['Life satisfaction'].values # Reshape to 1D array  

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X, y)

# Predict life satisfaction for a new GDP per capita value
X_new = np.array([[37_655.2]])
y_pred = knn.predict(X_new)
print(f"Predicted life satisfaction for GDP per capita of {X_new[0][0]}: {y_pred[0]:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Grafiklerde kullanılacak veriler
data = {'Yıl': [1998, 1999],
        'Fiyat': [80000, 82000]}
df = pd.DataFrame(data)

# Grafik 1: Y ekseni 70000'den başlıyor
plt.figure(figsize=(8, 6))
sns.barplot(x='Yıl', y='Fiyat', data=df, palette=['#1f77b4', '#ff7f0e']) # Farklı renkler
plt.ylim(78000, 88000) # Y ekseni limitleri (biraz pay bırakarak)
plt.title('EV Fiyatı\nEv fiyatlarında büyük artış')
plt.ylabel('Fiyat')
plt.grid(axis='y', linestyle='--') # Yatay grid çizgileri
# Değerleri çubukların üzerine yazma
for index, row in df.iterrows():
    plt.text(index, row.Fiyat + 100, str(row.Fiyat), color='black', ha="center") # Değerleri biraz yukarıda göster
plt.savefig('grafik1_ev_fiyati_buyuk_artis.png')
plt.show()

# Grafik 2: Y ekseni 0'dan başlıyor
plt.figure(figsize=(8, 6))
sns.barplot(x='Yıl', y='Fiyat', data=df, palette=['#1f77b4', '#ff7f0e']) # Farklı renkler
plt.ylim(0, 88000) # Y ekseni limitleri (biraz pay bırakarak)
plt.title('EV Fiyatı\nEv fiyatları geçen yıla göre artış göstermiştir')
plt.ylabel('Fiyat')
plt.grid(axis='y', linestyle='--') # Yatay grid çizgileri
# Değerleri çubukların üzerine yazma
for index, row in df.iterrows():
    plt.text(index, row.Fiyat + 1000, str(row.Fiyat), color='black', ha="center") # Değerleri biraz yukarıda göster
plt.savefig('grafik2_ev_fiyati_oransal_artis.png')
plt.show()


df = sns.load_dataset('iris')
df.head()

from sklearn.linear_model import LogisticRegression

X = df[['sepal_length', 'sepal_width']].values
y = df['species'].values

# Encode the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Fit the logistic regression model
model = LogisticRegression()
model.fit(X, y_encoded)
# Predict the species for a new sample
X_new = np.array([[5.0, 3.5]])
y_pred = model.predict(X_new)
predicted_species = le.inverse_transform(y_pred)
print(f"Predicted species for sepal length 5.0 and sepal width 3.5: {predicted_species[0]}")

df = sns.load_dataset('titanic')
df.head()
# Fill missing values in 'age' column with the mean age
df['age'].fillna(df['age'].mean(), inplace=True)

df['age'].isnull().sum()  # Check if there are any missing values left
df['age'].hist(bins=30, edgecolor='black')
plt.title('Age Distribution of Titanic Passengers') 
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# box plot for age
df.boxplot(column='age', by='survived')
plt.title('Age Distribution by Survival Status')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()