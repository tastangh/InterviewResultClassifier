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

def save_training_log(epoch, training_losses, validation_losses):
    """
    Eğitim sürecindeki ara sonuçları zaman damgalı bir .txt dosyasına kaydeder.
    
    Args:
    epoch -- epoch sayısı (int)
    training_losses -- eğitim kayıpları listesi (list of float)
    validation_losses -- doğrulama kayıpları listesi (list of float)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"results/train_log_{timestamp}.txt"

    # Log dosyasını yazma
    with open(log_path, "w") as f:
        f.write("Epoch\tTraining Loss\tValidation Loss\n")
        for e in range(epoch):
            f.write(f"{e+1}\t{training_losses[e]:.4f}\t{validation_losses[e]:.4f}\n")

def plot_loss_graph(training_losses, validation_losses):
    """
    Eğitim ve doğrulama kayıplarını gösteren bir grafik oluşturur ve zaman damgalı olarak kaydeder.
    
    Args:
    training_losses -- eğitim kayıpları listesi (list of float)
    validation_losses -- doğrulama kayıpları listesi (list of float)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/graphs/loss_graph_{timestamp}.png"

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
    Veriyi yükler, lojistik regresyon modelini eğitir, ara sonuçları kaydeder, eğitim/doğrulama 
    kayıplarını grafikte gösterir ve eğitilen modelin ağırlıklarını kaydeder.
    """
    # Veriyi yükle ve böl
    X, y = load_data("dataset/hw1Data.txt")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Sonuçları kaydetmek için gerekli dizinleri oluştur
    initialize_results_directory()

    # Modeli eğit
    model = LogisticRegressionSGD(learning_rate=0.0001, epochs=100)
    training_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val)

    # Eğitim tamamlandıktan sonra ağırlıkları kaydet
    model.save_weights("results/model_weights.json")

    # Ara sonuçları ve grafik kaydet
    save_training_log(len(training_losses), training_losses, validation_losses)
    plot_loss_graph(training_losses, validation_losses)

if __name__ == "__main__":
    train_model()
