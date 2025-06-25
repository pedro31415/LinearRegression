import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# pre-processameto
train = pd.read_csv("SVM/csv/train.csv")
test = pd.read_csv("SVM/csv/test.csv")

print(train.info())
print(test.info())

print(train['Embarked'].unique())

train['Age'].fillna(train["Age"].median(), inplace=True)
train['Embarked'].fillna(train["Embarked"].mode()[0], inplace=True)
test['Age'].fillna(train['Age'].median(), inplace=True)
test['Fare'].fillna(train['Fare'].median(), inplace=True)

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train[features]
y = train['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Processamento

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = SVC(kernel='rbf', C=100, gamma=0.01)
model.fit(X_train, y_train)

# ======== SEM PCA ========
pred_orig = model.predict(X_val)
acc_orig = accuracy_score(y_val, pred_orig)
prec_orig = precision_score(y_val, pred_orig)
rec_orig = recall_score(y_val, pred_orig)
f1_orig = f1_score(y_val, pred_orig)

# ======== COM PCA ========
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)

model_pca = SVC(kernel='rbf', C=1, gamma='scale')
model_pca.fit(X_train_pca, y_train)
pred_pca = model_pca.predict(X_val_pca)
acc_pca = accuracy_score(y_val, pred_pca)
prec_pca = precision_score(y_val, pred_pca)
rec_pca = recall_score(y_val, pred_pca)
f1_pca = f1_score(y_val, pred_pca)

# Pos processamento
print()
print("Acurácia sem PCA:", acc_orig)
print("Precisão sem PCA:", prec_orig)
print("Recall sem PCA:", rec_orig)
print("F1-score sem PCA:", f1_orig)

print()
print("Acurácia com PCA:", acc_pca)
print("Precisão com PCA:", prec_pca)
print("Recall com PCA:", rec_pca)
print("F1-score com PCA:", f1_pca)

plt.figure(figsize=(16, 10))

# Visualização PCA (2D)
plt.subplot(2, 3, 1)
plt.scatter(X_val_pca[:, 0], X_val_pca[:, 1], c=y_val, cmap='coolwarm', edgecolors='k')
plt.title('Distribuição PCA (validação)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')

# Comparação de acurácia
plt.subplot(2, 3, 2)
plt.bar(['Sem PCA', 'Com PCA'], [acc_orig, acc_pca], color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.title('Comparação de Acurácia')
plt.ylabel('Acurácia')

# Comparação de precisão
plt.subplot(2, 3, 3)
plt.bar(['Sem PCA', 'Com PCA'], [prec_orig, prec_pca], color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.title('Comparação de Precisão')
plt.ylabel('Precisão')

# Comparação de recall
plt.subplot(2, 3, 4)
plt.bar(['Sem PCA', 'Com PCA'], [rec_orig, rec_pca], color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.title('Comparação de Recall')
plt.ylabel('Recall')

# Comparação de F1-score
plt.subplot(2, 3, 5)
plt.bar(['Sem PCA', 'Com PCA'], [f1_orig, f1_pca], color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.title('Comparação de F1-score')
plt.ylabel('F1-score')

plt.tight_layout()
plt.show()

# ======== Matrizes de Confusão ========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_val, pred_orig, ax=axes[0], cmap='Blues')
axes[0].set_title('Matriz de Confusão - Sem PCA')
ConfusionMatrixDisplay.from_predictions(y_val, pred_pca, ax=axes[1], cmap='Oranges')
axes[1].set_title('Matriz de Confusão - Com PCA')
plt.tight_layout()
plt.show()

X_test = test[features]
# Preencher NaNs em todas as colunas (numéricas e categóricas)
for col in X_test.columns:
    if X_test[col].isnull().any():
        if X_test[col].dtype == 'float64' or X_test[col].dtype == 'int64':
            X_test[col].fillna(X_test[col].median(), inplace=True)
        else:
            X_test[col].fillna(X_test[col].mode()[0], inplace=True)
# Verificar se ainda existe algum NaN
assert not X_test.isnull().any().any(), f"Ainda existem NaNs em: {X_test.columns[X_test.isnull().any()].tolist()}"
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('SVM/csv/submission.csv', index=False)

