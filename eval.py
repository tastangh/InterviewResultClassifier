import numpy as np
import os

class Evaluator:
    def __init__(self, model, log_dir="results"):
        self.model = model
        self.log_dir = log_dir

    def confusion_matrix(self, y_target, y_pred, dataset_name="dataset"):
        y_target = y_target.astype(int)
        y_pred = np.array(y_pred).astype(int)
        
        TP = np.sum((y_target == 1) & (y_pred == 1))
        TN = np.sum((y_target == 0) & (y_pred == 0))
        FP = np.sum((y_target == 0) & (y_pred == 1))
        FN = np.sum((y_target == 1) & (y_pred == 0))
        
        # Confusion matrix değerlerini ve dataset adını ekrana yazdır
        print(f"\nConfusion Matrix for {dataset_name}:")
        print(f"TP (True Positive): {TP}")
        print(f"TN (True Negative): {TN}")
        print(f"FP (False Positive): {FP}")
        print(f"FN (False Negative): {FN}")
        
        # Gerçek ve tahmin değerlerini ekrana yazdır
        print(f"\nGerçek Değerler (y_target) for {dataset_name}:", y_target)
        print(f"Tahmin Değerleri (y_pred) for {dataset_name}:", y_pred)
        
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    def evaluate_metrics(self, y_target, y_pred, dataset_name="dataset"):
        conf_matrix = self.confusion_matrix(y_target, y_pred, dataset_name)
        TP, TN, FP, FN = conf_matrix["TP"], conf_matrix["TN"], conf_matrix["FP"], conf_matrix["FN"]
        
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        
        # Metrikleri ekrana yazdır
        print(f"\nMetrics for {dataset_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def evaluate(self, X, y, dataset_name="test"):
        y_pred = self.model.predict(X)
        metrics = self.evaluate_metrics(y, y_pred, dataset_name)
        return metrics

    def save_results(self, train_metrics, val_metrics, test_metrics, learning_rate, epochs):
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, f"eval_results_lr{learning_rate}_epochs{epochs}.txt")
        with open(log_path, "w") as f:
            f.write("Sonuçlar:\n")
            f.write(f"Learning Rate: {learning_rate}\n")
            f.write(f"Epochs: {epochs}\n\n")
            for dataset_name, metrics in [("Eğitim Seti", train_metrics), ("Doğrulama Seti", val_metrics), ("Test Seti", test_metrics)]:
                f.write(f"{dataset_name}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n\n")
