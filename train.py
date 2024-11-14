import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from dataset import DataProcessor
from logistic_model import LogisticRegressionModel

class Trainer:
    """
    Model eğitimi sürecini yöneten sınıf. Eğitim sürecinde kayıpları loglar, kayıp grafiği oluşturur 
    ve eğitilmiş modeli kaydeder.

    Özellikler:
        learning_rate (float): Modeli eğitmek için kullanılan öğrenme oranı.
        epochs (int): Modelin eğitimde geçireceği epoch sayısı.
        model (LogisticRegressionModel): Eğitimde kullanılan lojistik regresyon modeli.
        results_dir (str): Eğitim sonuçlarının kaydedileceği klasör yolu.
        model_dir (str): Eğitilen modelin kaydedileceği model klasör yolu.
    """

    def __init__(self, learning_rate=0.001, epochs=5000):
        """
        Trainer sınıfı, eğitim sürecini yönetir ve sonuçları kaydeder.
        
        Args:
            learning_rate (float): Öğrenme oranı.
            epochs (int): Eğitim süresince yinelenecek epoch sayısı.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = LogisticRegressionModel(learning_rate=self.learning_rate, epochs=self.epochs)
        self.results_dir = f"results/lr_{self.learning_rate}_epochs_{self.epochs}"
        self.model_dir = os.path.join(self.results_dir, "model")

    def initialize_results_directory(self):
        """
        Eğitim sonuçları ve modelin kaydedileceği dizinleri oluşturur.
        """
        os.makedirs(self.model_dir, exist_ok=True)

    def save_training_log(self, epoch, training_losses, validation_losses):
        """
        Her epoch için eğitim ve doğrulama kayıplarını bir log dosyasına kaydeder.

        Args:
            epoch (int): Epoch sayısı.
            training_losses (list of float): Eğitim kayıpları listesi.
            validation_losses (list of float): Doğrulama kayıpları listesi.
        """
        log_path = os.path.join(self.results_dir, f"train_log_lr_{self.learning_rate}_epochs_{self.epochs}.txt")
        with open(log_path, "w") as f:
            f.write("Epoch\tTraining Loss\tValidation Loss\n")
            for e in range(epoch):
                f.write(f"{e+1}\t\t{training_losses[e]:.4f}\t\t{validation_losses[e]:.4f}\n")

    def plot_loss_graph(self, training_losses, validation_losses):
        """
        Eğitim ve doğrulama kayıplarının grafiksel gösterimini oluşturur ve kaydeder.

        Args:
            training_losses (list of float): Eğitim kayıpları listesi.
            validation_losses (list of float): Doğrulama kayıpları listesi.
        """
        save_path = os.path.join(self.results_dir, f"loss_graph_lr_{self.learning_rate}_epochs_{self.epochs}.png")
        plt.figure(figsize=(10, 6))
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.title(f"Eğitim ve Doğrulama Kayıpları (lr: {self.learning_rate}, epochs: {self.epochs})")
        plt.savefig(save_path)
        plt.show()

    def save_model(self):
        """
        Eğitilmiş modeli belirtilen dizine kaydeder.
        """
        model_path = os.path.join(self.model_dir, f"logistic_model_lr_{self.learning_rate}_epochs_{self.epochs}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model '{model_path}' konumuna kaydedildi.")

    def train(self, file_path="dataset/hw1Data.txt"):
        """
        Veriyi yükleyip eğitim ve doğrulama setlerine ayırır, modeli eğitir, kayıp loglarını ve grafiğini oluşturur, 
        eğitilmiş modeli kaydeder.

        Args:
            file_path (str, opsiyonel): Eğitim verilerinin dosya yolu. Varsayılan 'dataset/hw1Data.txt'.

        Returns:
            tuple: (model, training_losses, validation_losses) - Eğitilmiş model, eğitim ve doğrulama kayıpları.
        """
        # Veriyi yükleyip eğitim, doğrulama ve test setlerine ayırma
        dataset = DataProcessor(file_path)
        X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
        
        # Sonuç dizinlerini oluştur
        self.initialize_results_directory()
        
        # Modeli eğit ve kayıpları kaydet
        training_losses, validation_losses = self.model.fit(X_train, y_train, X_val, y_val)
        
        # Eğitim loglarını ve grafikleri kaydet
        self.save_training_log(len(training_losses), training_losses, validation_losses)
        self.plot_loss_graph(training_losses, validation_losses)
        
        # Eğitilmiş modeli kaydet
        self.save_model()
        
        return self.model, training_losses, validation_losses

if __name__ == "__main__":
    """
    Ana program: Trainer sınıfını kullanarak modeli eğitir.
    - Veriyi yükleyip eğitim, doğrulama ve test setlerine ayırır.
    - Modeli belirtilen öğrenme oranı ve epoch sayısı ile eğitir.
    - Eğitim süreci sırasında kayıpları loglar ve kayıp grafiğini kaydeder.
    - Eğitilmiş modeli belirtilen dizine kaydeder.
    """
    learning_rate = 0.0001
    epochs = 5000
    data_path = "dataset/hw1Data.txt"
    
    # Trainer sınıfını başlat ve modeli eğit
    trainer = Trainer(learning_rate=learning_rate, epochs=epochs)
    model, training_losses, validation_losses = trainer.train(data_path)
    
    print("Eğitim tamamlandı. Eğitim logu, kayıp grafiği ve model 'results' dizininde kaydedildi.")
