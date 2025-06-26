import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIGURAÇÕES
INPUT_FOLDER = "Kmeans/imgs"
MAX_IMAGE_SIZE = (256, 256)
N_CLUSTERS = 2
NOISE_STDDEV = 0  #caso eu queira adicionar ruído pra testar o pca
def carregar_imagem(path):
    image = io.imread(path)
    image = resize(image, MAX_IMAGE_SIZE, anti_aliasing=True)
    image = (image * 255).astype(np.uint8)
    return image

def adicionar_ruido_gaussiano(image, stddev=NOISE_STDDEV):
    ruido = np.random.normal(0, stddev, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + ruido
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def gerar_ground_truth(image):
    gray = color.rgb2gray(image)
    limiar = np.mean(gray)
    mask = (gray > limiar).astype(int).reshape(-1)
    return mask

def extrair_features(image):
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    features = np.dstack([
        image[:, :, 0],  # R
        image[:, :, 1],  # G
        image[:, :, 2],  # B
        X, Y
    ])
    return features.reshape(-1, 5)

def ajustar_rotulo(true_labels, pred_labels):
    return pred_labels if accuracy_score(true_labels, pred_labels) >= accuracy_score(true_labels, 1 - pred_labels) else 1 - pred_labels

def segmentar_kmeans(features, usar_pca=False):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    if usar_pca:
        features_scaled = PCA(n_components=3).fit_transform(features_scaled)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    return kmeans.fit_predict(features_scaled)

def plotar_matriz_confusao(cm, ax, titulo):
    ax.imshow(cm, cmap="Blues", interpolation='nearest')
    ax.set_title(titulo)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Classe 0', 'Classe 1'])
    ax.set_yticklabels(['Classe 0', 'Classe 1'])
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")

def mostrar_resultados_unicos(image, ground_truth, labels_sem_pca, labels_com_pca, nome_img):
    h, w = image.shape[:2]
    cm_sem_pca = confusion_matrix(ground_truth, labels_sem_pca)
    cm_com_pca = confusion_matrix(ground_truth, labels_com_pca)

    plt.figure(figsize=(18, 10))

    # Linha 1 - imagens
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title("Imagem Original + Ruído")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(ground_truth.reshape(h, w), cmap='gray')
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(labels_sem_pca.reshape(h, w), cmap='gray')
    plt.title("Segmentação KMeans sem PCA")
    plt.axis("off")

    # Linha 2 - segmentação e matrizes
    plt.subplot(2, 3, 4)
    plt.imshow(labels_com_pca.reshape(h, w), cmap='gray')
    plt.title("Segmentação KMeans com PCA")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plotar_matriz_confusao(cm_sem_pca, plt.gca(), "Matriz Confusão - Sem PCA")

    plt.subplot(2, 3, 6)
    plotar_matriz_confusao(cm_com_pca, plt.gca(), "Matriz Confusão - Com PCA")

    plt.suptitle(f"Resultados - {nome_img}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def main():
    imagens = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imagens:
        print("Nenhuma imagem encontrada.")
        return

    for nome_img in imagens:
        print(f"\nProcessando: {nome_img}")
        path = os.path.join(INPUT_FOLDER, nome_img)

        image = carregar_imagem(path)
        image_ruidosa = adicionar_ruido_gaussiano(image)
        ground_truth = gerar_ground_truth(image)
        features = extrair_features(image_ruidosa)

        labels_kmeans = segmentar_kmeans(features, usar_pca=False)
        labels_pca = segmentar_kmeans(features, usar_pca=True)

        ajust_kmeans = ajustar_rotulo(ground_truth, labels_kmeans)
        ajust_pca = ajustar_rotulo(ground_truth, labels_pca)

        print("Sem PCA:")
        print(classification_report(ground_truth, ajust_kmeans))
        print("Com PCA:")
        print(classification_report(ground_truth, ajust_pca))

        mostrar_resultados_unicos(image_ruidosa, ground_truth, ajust_kmeans, ajust_pca, nome_img)

if __name__ == "__main__":
    main()
