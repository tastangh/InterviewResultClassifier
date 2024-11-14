import numpy as np
import os
import pickle
from dataset import DataProcessor

class Evaluator:
    def __init__(self, model=None, learning_rate=0.001, epochs=5000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.results_dir = f"results/lr_{self.learning_rate}_epochs_{self.epochs}"
        
    def load_model(self, model_path):
        """Belirtilen yolda model dosyası varsa modeli yükler."""
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Model '{model_path}' başarıyla yüklendi.")
        else:
            raise FileNotFoundError(f"'{model_path}' konumunda model bulunamadı. Lütfen yolu kontrol edin veya önce modeli eğitin.")

    def confusion_matrix(self, y_target, y_pred, dataset_name="dataset"):
        y_target = y_target.astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        TP = np.sum((y_target == 1) & (y_pred == 1))
        TN = np.sum((y_target == 0) & (y_pred == 0))
        FP = np.sum((y_target == 0) & (y_pred == 1))
        FN = np.sum((y_target == 1) & (y_pred == 0))
        
        print(f"\n{dataset_name} için Confusion Matrix:")
        print(f"TP (True Positive): {TP}")
        print(f"TN (True Negative): {TN}")
        print(f"FP (False Positive): {FP}")
        print(f"FN (False Negative): {FN}")

        # Gerçek ve tahmin değerlerini ekrana yazdır
        print(f"\nGerçek Değerler (y_target) for {dataset_name}:", y_target)
        print(f"Tahmin Değerleri (y_pred) for {dataset_name}: ", y_pred)
        
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    def evaluate_metrics(self, y_target, y_pred, dataset_name="dataset"):
        conf_matrix = self.confusion_matrix(y_target, y_pred, dataset_name)
        TP, TN, FP, FN = conf_matrix["TP"], conf_matrix["TN"], conf_matrix["FP"], conf_matrix["FN"]
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) 
        precision = TP / (TP + FP) 
        recall = TP / (TP + FN) 
        f1_score = 2 * (precision * recall) / (precision + recall)
        # Sonuçları döndür
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

    def evaluate(self, X, y, dataset_name="test"):
        if self.model is None:
            raise ValueError("No model is loaded. Please load a model before evaluation.")
        y_pred = self.model.predict(X)
        metrics = self.evaluate_metrics(y, y_pred, dataset_name)
        return metrics

    def save_results(self, train_metrics, val_metrics, test_metrics):
        os.makedirs(self.results_dir, exist_ok=True)
        log_path = os.path.join(self.results_dir, f"eval_results_lr_{self.learning_rate}_epochs_{self.epochs}.txt")
        with open(log_path, "w") as f:
            f.write("Değerlendirme Sonuçları:\n")
            f.write(f"Learning Rate: {self.learning_rate}\n")
            f.write(f"Epochs: {self.epochs}\n\n")
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
    
    learning_rate = 0.0001
    epochs = 5000
    model_path = f"results/lr_{learning_rate}_epochs_{epochs}/model/logistic_model_lr_{learning_rate}_epochs_{epochs}.pkl"
    data_path = "dataset/hw1Data.txt"
    
    evaluator = Evaluator(learning_rate=learning_rate, epochs=epochs)
    try:
        evaluator.load_model(model_path)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    dataset = DataProcessor(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
    
    train_metrics = evaluator.evaluate(X_train, y_train, "Eğitim Seti")
    val_metrics = evaluator.evaluate(X_val, y_val, "Doğrulama Seti")
    test_metrics = evaluator.evaluate(X_test, y_test, "Test Seti")
    
    evaluator.save_results(train_metrics, val_metrics, test_metrics)
    print("Değerlendirme tamamlandı. Sonuçlar 'results' dizininde kaydedildi.")
