import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from dataset import DataProcessor

class Evaluator:
    def __init__(self, model=None, learning_rate=0.001, epochs=5000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.results_dir = f"results/lr_{self.learning_rate}_epochs_{self.epochs}"
        self.model_path = f"{self.results_dir}/model/logistic_model_lr_{self.learning_rate}_epochs_{self.epochs}.pkl"
        
    def load_model(self):
        """Loads the model from the specified path if it exists."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"'{self.model_path}' yolunda model bulunamadı. Lütfen yolu kontrol edin veya modeli önce eğitin.")
    
    def confusion_matrix(self, y_target, y_pred, dataset_name="dataset"):
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
        # Confusion matrix array with respective labels
        matrix = np.array([[TP, FN], [FP, TN]])  # Adjusted for new axis arrangement
        
        fig, ax = plt.subplots()
        ax.matshow(matrix, cmap="Blues", alpha=0.7)
        
        # Annotating each cell with the count and label
        labels = [["TP", "FN"], ["FP", "TN"]]
        for (i, j), val in np.ndenumerate(matrix):
            label = labels[i][j]
            ax.text(j, i, f"{label}: {val}", ha="center", va="center", fontsize=12)
        
        # Setting x and y axis labels to the specified order
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Tahmin 1", "Tahmin 0"])
        ax.set_yticklabels(["Gerçek 1", "Gerçek 0"])
        
        # Adding axis titles
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        plt.title(f"Confusion Matrix - {dataset_name}")
        
        # Saving the modified confusion matrix plot
        os.makedirs(self.results_dir, exist_ok=True)
        plt.savefig(os.path.join(self.results_dir, f"confusion_matrix_{dataset_name}_lr_{self.learning_rate}_epochs_{self.epochs}.png"))
        plt.close()
        
    def evaluate_metrics(self, y_target, y_pred, dataset_name="veriseti"):
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
        # Ensure the model is loaded
        try:
            self.load_model()
        except FileNotFoundError as e:
            print(e)
            raise 

        # Proceed with evaluation if model is loaded successfully
        y_pred = self.model.predict(X)
        metrics = self.evaluate_metrics(y, y_pred, dataset_name)
        return metrics

    def save_results(self, train_metrics, val_metrics, test_metrics):
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
    learning_rate = 0.0001
    epochs = 5000
    data_path = "dataset/hw1Data.txt"

    evaluator = Evaluator(learning_rate=learning_rate, epochs=epochs)

    dataset = DataProcessor(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
    
    train_metrics = evaluator.evaluate(X_train, y_train, "Eğitim")
    val_metrics = evaluator.evaluate(X_val, y_val, "Doğrulama")
    test_metrics = evaluator.evaluate(X_test, y_test, "Test")
    
    if train_metrics and val_metrics and test_metrics:
        evaluator.save_results(train_metrics, val_metrics, test_metrics)
        print("Değerlendirme tamamlandı. Sonuçlar 'results' klasörüne kaydedildi.")