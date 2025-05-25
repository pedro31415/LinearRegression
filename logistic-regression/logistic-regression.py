import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar o dataset
df = pd.read_csv('logistic-regression/csv/emails.csv', sep=',', header=None, names=['text', 'spam'])

# Verificar dados
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df['spam'].value_counts())
# print(df.tail())


# Pre-processamento

# Codificar os rótulos
df['spam'] = df['spam'].map({'ham': 0, 'spam': 1})

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray()
y = df['spam'].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Processamento

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights):
    m = len(y)
    h = sigmoid(np.dot(X, weights))
    cost = - (1/m) * np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))  # evitar log(0)
    return cost

def gradient_descent(X, y, weights, lr, epochs):
    m = len(y)
    for i in range(epochs):
        h = sigmoid(np.dot(X, weights))
        gradient = np.dot(X.T, (h - y)) / m
        weights -= lr * gradient
        
        if i % 100 == 0:
            cost = compute_cost(X, y, weights)
            print(f"Epoch {i}, Cost: {cost:.4f}")
    
    return weights

# Adicionar bias (coluna de 1s)
X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Inicializar pesos
weights = np.zeros(X_train_bias.shape[1])

# Treinar o modelo
weights = gradient_descent(X_train_bias, y_train, weights, lr=0.1, epochs=100)

# Pós-processamento

def predict(X, weights, threshold=0.5):
    probs = sigmoid(np.dot(X, weights))
    return (probs >= threshold).astype(int)

# Fazer previsões
y_pred = predict(X_test_bias, weights)

# Avaliação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Acurácia:", accuracy_score(y_test, y_pred))
print("Precisão:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))



