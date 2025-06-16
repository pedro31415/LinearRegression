import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping

# --- 1. Configurações Iniciais ---
# Caminho para a pasta com as imagens.
# O script espera que a pasta UTKFace esteja no mesmo diretório.
dataset_path = 'RNA/UTKFace'
if not os.path.isdir(dataset_path):
    print(f"Erro: O diretório do dataset '{dataset_path}' não foi encontrado.")
    print("Por favor, baixe o dataset e coloque a pasta 'UTKFace' no mesmo local que este script.")
    exit()

image_files = os.listdir(dataset_path)

# Dimensões para redimensionar as imagens
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Limitar o número de imagens para um teste mais rápido (opcional).
# Para usar o dataset completo, defina como None.
MAX_IMAGES = 8000
if MAX_IMAGES:
    image_files = image_files[:MAX_IMAGES]

# --- 2. Carregamento e Pré-processamento dos Dados ---
images = []
labels = []

print(f"Iniciando o carregamento de até {len(image_files)} imagens...")

for i, filename in enumerate(image_files):
    if not filename.lower().endswith(('.jpg', '.jpeg')):
        continue
    
    try:
        # Extrai o gênero (0=Homem, 1=Mulher) do nome do arquivo
        parts = filename.split('_')
        if len(parts) < 2:
            continue
        gender = int(parts[1])
        
        img_path = os.path.join(dataset_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Aviso: Não foi possível carregar a imagem {filename}. Pulando.")
            continue
            
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        images.append(img)
        labels.append(gender)
    except (ValueError, IndexError):
        # Ignora arquivos com nome fora do padrão
        continue

print("Carregamento e pré-processamento concluídos.")

# Converte listas para arrays NumPy e normaliza os pixels da imagem (0 a 1)
images = np.array(images, dtype='float32') / 255.0
labels = np.array(labels, dtype='int32')

print(f"\nFormato do array de imagens: {images.shape}")
print(f"Formato do array de labels: {labels.shape}")

# Divide os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Construção do Modelo (CNN)
model = Sequential([
    Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Técnica para evitar overfitting
    Dense(1, activation='sigmoid') # Camada de saída para classificação binária
])

print("\nResumo do Modelo:")
model.summary()

# Compilação do Modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Treinamento da Rede Neural
# Callback para parar o treino se a performance não melhorar
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nIniciando o treinamento...")
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=2
)

# Avaliação do Modelo
print("\nAvaliando o modelo...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_accuracy:.4f}")
print(f"Perda no conjunto de teste: {test_loss:.4f}")

# Plotando os resultados do treinamento
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo-', label='Acurácia de Treino')
plt.plot(epochs_range, val_acc, 'ro-', label='Acurácia de Validação')
plt.title('Acurácia de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo-', label='Perda de Treino')
plt.plot(epochs_range, val_loss, 'ro-', label='Perda de Validação')
plt.title('Perda de Treino e Validação')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()
plt.grid(True)

plt.suptitle('Resultados do Treinamento do Modelo')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('RNA/imgs/training_results.png')

# plt.show() # Descomente para exibir o gráfico interativamente ao final
# Salva o modelo treinado
output_dir = 'RNA/models'
model_path = os.path.join(output_dir, 'gender_classifier.keras')
model.save(model_path)
print(f"Modelo salvo com sucesso em '{model_path}'")
