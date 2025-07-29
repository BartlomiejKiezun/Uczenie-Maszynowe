# naive_bayes_spam.py

# Importujemy biblioteki
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # Do przekształcenia tekstu na liczby
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes do tekstu
from sklearn.metrics import accuracy_score, classification_report

#Wczytanie danych
df = pd.read_csv('zbiory/spam.csv', encoding='latin-1')

# Usuwamy zbędne kolumny
df = df[['v1', 'v2']]
df.columns = ['label', 'message']  # Zmieniamy nazwy kolumn



# Konwersja etykiet na liczby
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Podział na X  i y
X = df['message']
y = df['label_num']

# zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#  Przekształcenie tekstu na liczby
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Naive Bayes do tekstu
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Przewidywanie
y_pred = model.predict(X_test_counts)

print("\n Dokładność:")
print(accuracy_score(y_test, y_pred))

print("\n Raport :")
print(classification_report(y_test, y_pred))
