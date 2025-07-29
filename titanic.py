# naive_bayes_titanic.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt  #

#  Wczytanie danych
df = pd.read_csv('zbiory/train.csv')

#  kolumny
df = df[['Survived', 'Pclass', 'Sex', 'Age']].dropna()
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

X = df[['Pclass', 'Sex', 'Age']]
y = df['Survived']

# Podział
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model
model = GaussianNB()
model.fit(X_train, y_train)

#  Predykcja
y_pred = model.predict(X_test)

print("Dokładność:", accuracy_score(y_test, y_pred))
print("\nRaport:\n", classification_report(y_test, y_pred))

#  Wykres
df['Survived'].value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Titanic: ')
plt.xlabel('0 = Nie przeżył, 1 = Przeżył')
plt.ylabel('Liczba pasażerów')
plt.xticks(rotation=0)
plt.show()
