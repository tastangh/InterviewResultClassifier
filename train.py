import matplotlib.pyplot as plt
import os
from datetime import datetime
from dataset import load_data, split_data
from logistic_model import LogisticRegressionSGD

def initialize_results_directory():
    """
    Sonuçları kaydetmek için gerekli klasörleri oluşturur.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)

def save_training_log(epoch, training_losses, validation_losses, learning_rate, total_epochs):
    """
    Eğitim sürecindeki ara sonuçları öğrenme oranı ve toplam epoch bilgisi ile kaydeder.
    
    Argümanlar:
    epoch -- epoch sayısı (int)
    training_losses -- eğitim kayıpları listesi (list of float)
    validation_losses -- doğrulama kayıpları listesi (list of float)
    learning_rate -- modelin öğrenme oranı (float)
    total_epochs -- toplam epoch sayısı (int)
    """
    # Dosya adında öğrenme oranı ve epoch bilgisi ekleyin
    log_path = f"results/train_log_lr{learning_rate}_epochs{total_epochs}.txt"

    # Log dosyasını yazma
    with open(log_path, "w") as f:
        f.write("Epoch\tTraining Loss\tValidation Loss\n")
        for e in range(epoch):
            f.write(f"{e+1}\t{training_losses[e]:.4f}\t{validation_losses[e]:.4f}\n")

def plot_loss_graph(training_losses, validation_losses, learning_rate, total_epochs):
    """
    Eğitim ve doğrulama kayıplarını gösteren bir grafik oluşturur ve öğrenme oranı ve epoch bilgisi ile kaydeder.
    
    Argümanlar:
    training_losses -- eğitim kayıpları listesi (list of float)
    validation_losses -- doğrulama kayıpları listesi (list of float)
    learning_rate -- modelin öğrenme oranı (float)
    total_epochs -- toplam epoch sayısı (int)
    """
    # Dosya adında öğrenme oranı ve epoch bilgisi ekleyin
    save_path = f"results/graphs/loss_graph_lr{learning_rate}_epochs{total_epochs}.png"

    # Grafik oluşturma ve kaydetme
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.title("Eğitim ve Doğrulama Kayıpları")
    plt.savefig(save_path)
    plt.show()

def train_model():
    """
    Veriyi yükler, lojistik regresyon modelini eğitir, ara sonuçları kaydeder ve eğitim/doğrulama 
    kayıplarını grafikte gösterir.
    """
    # Veriyi yükle ve böl
    X, y = load_data("dataset/hw1Data.txt")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Sonuçları kaydetmek için gerekli dizinleri oluştur
    initialize_results_directory()

    # Model parametreleri
    learning_rate = 0.0005  # Learning rate'i düşürdük
    epochs = 1000  # Epoch sayısını artırdık

    # Modeli eğit
    model = LogisticRegressionSGD(learning_rate=learning_rate, epochs=epochs)
    training_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val)

    # Ara sonuçları ve grafik kaydet
    save_training_log(len(training_losses), training_losses, validation_losses, learning_rate, epochs)
    plot_loss_graph(training_losses, validation_losses, learning_rate, epochs)

if __name__ == "__main__":
    train_model()
