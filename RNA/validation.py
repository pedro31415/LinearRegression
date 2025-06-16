import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURAÇÕES ---
# Defina os caminhos e parâmetros.
MODEL_PATH = 'RNA/models/gender_classifier.keras'
DATASET_PATH = 'RNA/UTKFace'
IMG_HEIGHT = 128
IMG_WIDTH = 128
MAX_IMAGES = 8000 

# --- 2. FUNÇÕES AUXILIARES ---

def load_and_preprocess_data(dataset_path, max_images):
    """
    Carrega o dataset UTKFace, extraindo imagens, rótulos de gênero e idade.
    Esta função é necessária para obter o conjunto de teste original.
    """
    if not os.path.isdir(dataset_path):
        print(f"Erro: Diretório do dataset '{dataset_path}' não encontrado.")
        return None, None, None

    print(f"Carregando dados de '{dataset_path}'...")
    images, genders, ages = [], [], []
    image_files = os.listdir(dataset_path)
    if max_images:
        image_files = image_files[:max_images]

    for filename in image_files:
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        try:
            parts = filename.split('_')
            if len(parts) < 3: continue
            
            age = int(parts[0])
            gender = int(parts[1])
            
            img_path = os.path.join(dataset_path, filename)
            img = cv2.imread(img_path)
            if img is None: continue
            
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            images.append(img)
            genders.append(gender)
            ages.append(age)
        except (ValueError, IndexError):
            continue
            
    # Normaliza as imagens para o intervalo [0, 1]
    images = np.array(images, dtype='float32') / 255.0
    genders = np.array(genders, dtype='int32')
    ages = np.array(ages, dtype='int32')
    
    print("Carregamento concluído.")
    return images, genders, ages

def plot_dataset_distributions(genders, ages):
    """
    Cria e exibe gráficos sobre a distribuição dos dados base.
    """
    print("Gerando gráficos de distribuição do dataset base...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Distribuição de Gênero
    gender_counts = np.bincount(genders)
    gender_labels = ['Homem (0)', 'Mulher (1)']
    ax1.bar(gender_labels, gender_counts, color=['#3498db', '#e74c3c'])
    ax1.set_title('Distribuição de Gênero no Dataset', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Quantidade de Imagens', fontsize=12)
    for i, count in enumerate(gender_counts):
        ax1.text(i, count + 50, str(count), ha='center', fontsize=12)

    # Gráfico 2: Distribuição de Idade
    ax2.hist(ages, bins=30, color='#2ecc71', edgecolor='black')
    ax2.set_title('Distribuição de Idade no Dataset', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Idade', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)

    plt.suptitle('📊 Análise Exploratória do Dataset UTKFace', fontsize=18, y=1)
    plt.tight_layout()
    plt.show()

def plot_validation_results(y_true, y_pred_classes):
    """
    Exibe o relatório de classificação e a matriz de confusão.
    """
    print("\n--- Relatório de Classificação ---")
    report = classification_report(y_true, y_pred_classes, target_names=['Homem (0)', 'Mulher (1)'])
    print(report)

    print("Gerando Matriz de Confusão...")
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Homem', 'Mulher'], 
                yticklabels=['Homem', 'Mulher'],
                annot_kws={"size": 16})
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.ylabel('Verdadeiro (Real)', fontsize=12)
    plt.xlabel('Previsto (Modelo)', fontsize=12)
    plt.show()

def plot_example_predictions(model, X_test, y_test, num_examples=10):
    """
    Mostra exemplos de previsões do modelo em imagens de teste.
    """
    print("\n--- Exemplos de Previsões do Modelo ---")
    # Seleciona exemplos aleatórios
    indices = np.random.choice(range(len(X_test)), size=num_examples, replace=False)
    
    plt.figure(figsize=(20, 8))
    for i, idx in enumerate(indices):
        ax = plt.subplot(2, 5, i + 1)
        img = X_test[idx]
        true_label = y_test[idx]
        
        # Faz a previsão
        prediction_prob = model.predict(np.expand_dims(img, axis=0))[0][0]
        predicted_label = 1 if prediction_prob > 0.5 else 0
        
        # Define o título e a cor
        is_correct = (predicted_label == true_label)
        title_color = 'green' if is_correct else 'red'
        label_map = {0: 'Homem', 1: 'Mulher'}
        
        title = f"Previsto: {label_map[predicted_label]}\nReal: {label_map[true_label]}"
        
        # Plota a imagem
        ax.imshow(img)
        ax.set_title(title, color=title_color, fontsize=14, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()


# --- 3. EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # Carrega e pré-processa todos os dados
    all_images, all_genders, all_ages = load_and_preprocess_data(DATASET_PATH, MAX_IMAGES)

    if all_images is not None:
        # 📊 Etapa de Análise Exploratória
        plot_dataset_distributions(all_genders, all_ages)

        # Recria a mesma divisão treino/teste para obter o X_test e y_test corretos
        # É crucial usar o mesmo random_state do script de treino!
        _, X_test, _, y_test = train_test_split(
            all_images, all_genders, test_size=0.2, random_state=42, stratify=all_genders
        )

        # 🤖 Etapa de Carregamento e Validação do Modelo
        if not os.path.exists(MODEL_PATH):
            print(f"Erro: Modelo '{MODEL_PATH}' não encontrado. Execute o script de treino primeiro.")
        else:
            print(f"\nCarregando modelo salvo de '{MODEL_PATH}'...")
            model = tf.keras.models.load_model(MODEL_PATH)
            model.summary()

            print("\nRealizando previsões no conjunto de teste...")
            predictions = model.predict(X_test)
            # Converte probabilidades (saída da sigmoid) para classes (0 ou 1)
            predicted_classes = (predictions > 0.5).astype("int32").flatten()

            # Gera as métricas e gráficos de validação
            plot_validation_results(y_test, predicted_classes)
            
            # Mostra exemplos visuais
            plot_example_predictions(model, X_test, y_test)