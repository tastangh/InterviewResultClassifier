import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from dataset import DataProcessor

class Evaluator:
    """
    Bir modeli çeşitli metrikler hesaplayarak değerlendirir ve confusion matrisi grafiğini oluşturur.

    Özellikler:
        model (object): Değerlendirilecek eğitilmiş model.
        learning_rate (float): Modeli eğitmek için kullanılan öğrenme oranı.
        epochs (int): Eğitimde kullanılan epoch sayısı.
        results_dir (str): Değerlendirme sonuçlarını ve modeli kaydetmek için klasör yolu.
        model_path (str): Kaydedilmiş model dosyasının yolu.
    """

    def __init__(self, model=None, learning_rate=0.001, epochs=5000):
        """
        Model, öğrenme oranı ve epoch ayarları ile Evaluator'ü başlatır.

        Args:
            model (object, opsiyonel): Değerlendirilecek model. Varsayılan None.
            learning_rate (float, opsiyonel): Öğrenme oranı. Varsayılan 0.001.
            epochs (int, opsiyonel): Epoch sayısı. Varsayılan 5000.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.results_dir = f"results/lr_{self.learning_rate}_epochs_{self.epochs}"
        self.model_path = f"{self.results_dir}/model/logistic_model_lr_{self.learning_rate}_epochs_{self.epochs}.pkl"
        
    def load_model(self):
        """
        Belirtilen yoldan modeli yükler, eğer dosya mevcut değilse hata verir.
        
        Raises:
            FileNotFoundError: Eğer model belirtilen yolda bulunamazsa.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"'{self.model_path}' yolunda model bulunamadı. Lütfen yolu kontrol edin veya modeli önce eğitin.")
    
    def confusion_matrix(self, y_target, y_pred, dataset_name="dataset"):
        """
        Hedef ve tahmin edilen değerlerle confusion matrisi hesaplar ve grafiğini çizer.
        
        Args:
            y_target (numpy.ndarray): Gerçek değerler.
            y_pred (numpy.ndarray): Tahmin edilen değerler.
            dataset_name (str, opsiyonel): Veriseti adı. Varsayılan 'dataset'.
        
        Returns:
            dict: confusion matrisi değerleri ("TP", "TN", "FP", "FN").
        """
        y_target = y_target.astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        TP = np.sum((y_target == 1) & (y_pred == 1))
        TN = np.sum((y_target == 0) & (y_pred == 0))
        FP = np.sum((y_target == 0) & (y_pred == 1))
        FN = np.sum((y_target == 1) & (y_pred == 0))
        
        print(f"\n{dataset_name} için Confusion Matrisi:")
        print(f"TP (True Positive): {TP}")
        print(f"TN (True Negative): {TN}")
        print(f"FP (False Positive): {FP}")
        print(f"FN (False Negative): {FN}")

        print(f"\nGerçek Değerler (y_target) for {dataset_name}:", y_target)
        print(f"Tahmin Değerleri (y_pred) for {dataset_name}: ", y_pred)

        self.plot_confusion_matrix(TP, TN, FP, FN, dataset_name)

        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    def plot_confusion_matrix(self, TP, TN, FP, FN, dataset_name):
        """
        confusion matrisi değerleri ile grafiğini çizer ve kaydeder.

        Args:
            TP (int): Doğru Pozitif sayısı.
            TN (int): Doğru Negatif sayısı.
            FP (int): Yanlış Pozitif sayısı.
            FN (int): Yanlış Negatif sayısı.
            dataset_name (str): Veriseti adı.
        """
        matrix = np.array([[TP, FN], [FP, TN]])
        
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap="Blues", alpha=0.7)
        
        labels = [["TP", "FN"], ["FP", "TN"]]
        for (i, j), val in np.ndenumerate(matrix):
            label = labels[i][j]
            ax.text(j, i, f"{label}: {val}", ha="center", va="center", fontsize=12)
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Tahmin 1", "Tahmin 0"])
        ax.set_yticklabels(["Gerçek 1", "Gerçek 0"])
        
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        plt.title(f"Confusion Matrix - {dataset_name}")
        
        os.makedirs(self.results_dir, exist_ok=True)
        plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{dataset_name}_lr_{self.learning_rate}_epochs_{self.epochs}.png"))
        plt.close()
        
    def evaluate_metrics(self, y_target, y_pred, dataset_name="veriseti"):
        """
        Belirtilen hedef ve tahmin edilen değerlerle metrikleri hesaplar.

        Args:
            y_target (numpy.ndarray): Gerçek değerler.
            y_pred (numpy.ndarray): Tahmin edilen değerler.
            dataset_name (str, opsiyonel): Veriseti adı. Varsayılan 'veriseti'.

        Returns:
            dict: Metrik değerleri ("accuracy", "precision", "recall", "f1_score").
        """
        conf_matrix = self.confusion_matrix(y_target, y_pred, dataset_name)
        TP, TN, FP, FN = conf_matrix["TP"], conf_matrix["TN"], conf_matrix["FP"], conf_matrix["FN"]
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def evaluate(self, X, y, dataset_name="test"):
        """
        Belirtilen veri üzerinde model performansını değerlendirir.

        Args:
            X (numpy.ndarray): Özellikler matrisi.
            y (numpy.ndarray): Etiketler.
            dataset_name (str, opsiyonel): Veriseti adı. Varsayılan 'test'.

        Returns:
            dict: Model performans metrikleri.
        
        Raises:
            FileNotFoundError: Model yüklenemezse.
        """
        try:
            self.load_model()
        except FileNotFoundError as e:
            print(e)
            raise 

        y_pred = self.model.predict(X)
        metrics = self.evaluate_metrics(y, y_pred, dataset_name)
        return metrics

    def save_results(self, train_metrics, val_metrics, test_metrics):
        """
        Eğitim, doğrulama ve test metrik sonuçlarını dosyaya kaydeder.

        Args:
            train_metrics (dict): Eğitim seti metrikleri.
            val_metrics (dict): Doğrulama seti metrikleri.
            test_metrics (dict): Test seti metrikleri.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        log_path = os.path.join(self.results_dir, f"eval_results_lr_{self.learning_rate}_epochs_{self.epochs}.txt")
        with open(log_path, "w") as f:
            f.write("Değerlendirme Sonuçları:\n")
            f.write(f"Öğrenme Oranı: {self.learning_rate}\n")
            f.write(f"Epoch Sayısı: {self.epochs}\n\n")
            for dataset_name, metrics in [
                ("Eğitim Seti", train_metrics),
                ("Doğrulama Seti", val_metrics),
                ("Test Seti", test_metrics)
            ]:
                f.write(f"{dataset_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")

if __name__ == "__main__":
    """
    Ana program: Modeli değerlendirir ve sonuçları kaydeder.
    - Öğrenme oranı ve epoch sayısı ile bir Evaluator nesnesi oluşturur.
    - Veriyi eğitim, doğrulama ve test setleri olarak böler.
    - Her bir veri seti için değerlendirme metriklerini hesaplar.
    - Sonuçları bir dosyaya kaydeder.
    """
    learning_rates = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs_list = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000,200000,500000]
    data_path = "dataset/hw1Data.txt"
    for learning_rate in learning_rates:
        for epochs in epochs_list:
            # Evaluator oluşturma
            evaluator = Evaluator(learning_rate=learning_rate, epochs=epochs)

            # Veriyi yükleyip eğitim, doğrulama ve test setlerine ayırma
            dataset = DataProcessor(data_path)
            X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
            
            # Her veri seti için değerlendirme
            train_metrics = evaluator.evaluate(X_train, y_train, "Eğitim")
            val_metrics = evaluator.evaluate(X_val, y_val, "Doğrulama")
            test_metrics = evaluator.evaluate(X_test, y_test, "Test")
            
            # Değerlendirme sonuçlarını kaydetme
            if train_metrics and val_metrics and test_metrics:
                evaluator.save_results(train_metrics, val_metrics, test_metrics)
                print("Değerlendirme tamamlandı. Sonuçlar 'results' klasörüne kaydedildi.")
