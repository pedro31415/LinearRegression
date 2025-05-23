import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("csv/student-por.csv", sep=";")


# Pré-processamento 
print(df.head())
print(df.describe())
print(df.info())

for col in df.columns:
    print(f"{col} - valores únicos: {df[col].nunique()}")
    print(df[col].value_counts())


# Verificar valores ausentes
print(df.isnull().sum())

# Se houverem poucos valores ausentes, podemos preencher:
df.fillna(df.mean(numeric_only=True), inplace=True)
sns.boxplot(data=df[['G1', 'G2', 'G3']])
# plt.title('Distribuição das Notas')
# plt.show()



#verificar se há outliers
# Selecionar apenas colunas numéricas (exceto a target "G3" por enquanto)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove('G3')


# Normalizar os dados
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


# codidificação de variáveis categóricas
df = pd.get_dummies(df, drop_first=True) 
print(df.info())

# dividir os dados em variáveis independentes (X) e dependentes (y)
X = df.drop("G3", axis=1).values
y = df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = X_train.astype(float)
y_train = y_train.astype(float)


# Processamento
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, y, lr=0.01, epochs=15000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # adiciona bias
    theta = np.random.randn(n + 1)  # inicializa pesos aleatoriamente
    
    for epoch in range(epochs):
        y_pred = X_b.dot(theta)
        error = y_pred - y
        gradients = (2/m) * X_b.T.dot(error)
        theta -= lr * gradients
        
        if epoch % 100 == 0:
            loss = mean_squared_error(y, y_pred)
            print(f"Epoch {epoch}, MSE: {loss:.4f}")
    
    return theta

# Treinar modelo
theta = gradient_descent(X_train, y_train)

# Fazer predições
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_pred = X_test_b.dot(theta)

# Pós-processamento
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")






