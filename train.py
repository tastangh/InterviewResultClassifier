# train.py
import matplotlib.pyplot as plt
import os
from dataset import DataProcessor
from logistic_model import LogisticRegressionModel

class Trainer:
    def __init__(self, learning_rate=0.001, epochs=5000):
        """
        Trainer sınıfı, eğitim sürecini yönetir ve sonuçları kaydeder.
        
        Args:
            learning_rate (float): Öğrenme oranı
            epochs (int): Eğitim süresince yinelenecek epoch sayısı
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = LogisticRegressionModel(learning_rate=self.learning_rate, epochs=self.epochs)

    def initialize_results_directory(self):
        os.makedirs("results", exist_ok=True)
        os.makedirs("results/graphs", exist_ok=True)

    def save_training_log(self, epoch, training_losses, validation_losses):
        log_path = f"results/train_log_lr{self.learning_rate}_epochs{self.epochs}.txt"
        with open(log_path, "w") as f:
            f.write("Epoch\tTraining Loss\tValidation Loss\n")
            for e in range(epoch):
                f.write(f"{e+1}\t\t{training_losses[e]:.4f}\t\t{validation_losses[e]:.4f}\n")

    def plot_loss_graph(self, training_losses, validation_losses):
        save_path = f"results/graphs/loss_graph_lr{self.learning_rate}_epochs{self.epochs}.png"
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.title(f"Eğitim ve Doğrulama Kayıpları (lr: {self.learning_rate}, epochs: {self.epochs})")
        plt.savefig(save_path)
        plt.show()

    def train(self, file_path="dataset/hw1Data.txt"):
        dataset = DataProcessor(file_path)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
        
        # Sonuç dizinlerini oluştur
        self.initialize_results_directory()
        
        # Modeli eğit ve kayıpları kaydet
        training_losses, validation_losses = self.model.fit(X_train, y_train, X_val, y_val)
        
        # Eğitim loglarını ve grafikleri kaydet
        self.save_training_log(len(training_losses), training_losses, validation_losses)
        self.plot_loss_graph(training_losses, validation_losses)
        
        return self.model, training_losses, validation_losses
