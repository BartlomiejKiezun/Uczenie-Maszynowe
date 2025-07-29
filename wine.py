# naive_bayes_wine.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt  # 📊

# 1️ Wczytanie danych
df = pd.read_csv('zbiory/winequality-red.csv')

X = df.drop('quality', axis=1)
y = df['quality']

# 2️ Podział
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3️ Model
model = GaussianNB()
model.fit(X_train, y_train)

# 4️ Predykcja
y_pred = model.predict(X_test)

print("Dokładność:", accuracy_score(y_test, y_pred))
print("\n Raport:\n", classification_report(y_test, y_pred))

# 5 Wykres —rozkład jakości wina
df['quality'].value_counts().sort_index().plot(kind='bar', color='purple')
plt.title('Rozkład jakości wina')
plt.xlabel('Jakość (ocena)')
plt.ylabel('Liczba próbek')
plt.xticks(rotation=0)
plt.show()
