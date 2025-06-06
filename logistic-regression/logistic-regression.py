import matplotlib
matplotlib.use('TkAgg') 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix
import scipy.sparse

# Carregar o dataset
df = pd.read_csv('logistic-regression/csv/emails.csv')

# Pré-processamento 

# Verificar tipos e valores únicos
print(df.dtypes)
print(df['spam'].unique())
print(df['spam'].value_counts())

# Verifica se há valores nulos
assert df['spam'].isnull().sum() == 0, "Existem valores nulos na coluna 'spam'."

# Vetorização TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['spam'].values

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Adicionar bias (coluna de 1s) corretamente para matrizes esparsas
X_train_bias = scipy.sparse.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_bias = scipy.sparse.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Verificações
assert not np.any(np.isnan(X_train_bias.data)), "X_train_bias contém NaN"
assert not np.any(np.isinf(X_train_bias.data)), "X_train_bias contém Inf"
assert not np.any(np.isnan(y_train)), "y_train contém NaN"
assert not np.any(np.isinf(y_train)), "y_train contém Inf"

# Processamento

# Sigmoid com clipping
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# Custo
def compute_cost(X, y, weights):
    m = len(y)
    z = X.dot(weights)
    if hasattr(z, "toarray"):
        z = z.toarray().flatten()
    h = sigmoid(z)
    h = np.clip(h, 1e-10, 1 - 1e-10)
    cost = - (1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) # Average Binary Cross-Entropy
    return cost

# Gradiente Descendente com registro dos custos
def gradient_descent(X, y, weights, lr, epochs):
    m = len(y)
    costs = []
    for i in range(epochs):
        z = X.dot(weights)
        if hasattr(z, "toarray"):
            z = z.toarray().flatten()
        h = sigmoid(z)
        gradient = X.T.dot(h - y) / m
        weights -= lr * gradient

        cost = compute_cost(X, y, weights)
        costs.append(cost)

        if i % 100 == 0:
            print(f"Epoch {i}, Cost: {cost:.4f}")

        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print(f"Pesos inválidos na época {i}")
            break
    return weights, costs

# Previsão com threshold ajustável
def predict(X, weights, threshold=0.3):  # Threshold menor para detectar mais spams
    z = X.dot(weights)
    if hasattr(z, "toarray"):
        z = z.toarray().flatten()
    probs = sigmoid(z)
    return (probs >= threshold).astype(int)

# Inicializar pesos
weights = np.zeros(X_train_bias.shape[1])

# Treinar modelo e armazenar custos
weights, costs = gradient_descent(X_train_bias, y_train, weights, lr=0.05, epochs=10000)

# Fazer previsões com threshold escolhido
chosen_threshold = 0.32
y_pred = predict(X_test_bias, weights, threshold=chosen_threshold)

# Pós-processamento

# Avaliação do modelo com zero_division=1 para evitar warnings
print("Acurácia:",(accuracy_score(y_test, y_pred) * 100))
print("Precisão:", (precision_score(y_test, y_pred, zero_division=1) * 100))
print("Recall:", (recall_score(y_test, y_pred, zero_division=1) * 100))
print("F1 Score:", (f1_score(y_test, y_pred, zero_division=1) * 100))

# Função para avaliar métricas em vários thresholds
def evaluate_thresholds(X, y_true, weights):
    thresholds = np.arange(0, 1.01, 0.01)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    z = X.dot(weights)
    if hasattr(z, "toarray"):
        z = z.toarray().flatten()
    probs = sigmoid(z)

    for thresh in thresholds:
        y_pred = (probs >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    return thresholds, accuracies, precisions, recalls, f1s

# Avaliar thresholds no conjunto de teste
thresholds, accuracies, precisions, recalls, f1s = evaluate_thresholds(X_test_bias, y_test, weights)

plt.figure(figsize=(6, 4))
sns.countplot(x='spam', data=df, palette='viridis')
plt.title('Distribuição das Classes (Spam vs Não Spam)')
plt.xlabel('Classe (0 = Não Spam, 1 = Spam)')
plt.ylabel('Quantidade')
plt.xticks([0, 1], ['Não Spam (0)', 'Spam (1)'])
plt.grid(axis='y')
plt.show()

# Plotar os gráficos das métricas para diferentes thresholds
plt.figure(figsize=(10,6))
plt.plot(thresholds, accuracies, label='Acurácia')
plt.plot(thresholds, precisions, label='Precisão')
plt.plot(thresholds, recalls, label='Recall')
plt.plot(thresholds, f1s, label='F1 Score')
plt.xlabel('Threshold')
plt.ylabel('Métrica')
plt.title('Métricas para diferentes thresholds')
plt.legend()
plt.grid(True)
plt.show()

# Plot custo durante o treinamento
plt.figure(figsize=(10,6))
plt.plot(range(len(costs)), costs, label='Custo')
plt.xlabel('Época')
plt.ylabel('Custo')
plt.title('Evolução do custo durante o treinamento')
plt.legend()
plt.grid(True)
plt.show()

# Curva ROC e AUC
z = X_test_bias.dot(weights)
if hasattr(z, "toarray"):
    z = z.toarray().flatten()
probs = sigmoid(z)

fpr, tpr, thresholds_roc = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0,1], [0,1], 'k--')  # linha diagonal baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Curva ROC')
plt.legend()
plt.grid(True)
plt.show()

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()
