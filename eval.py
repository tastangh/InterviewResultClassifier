import numpy as np
import os

class Evaluator:
    def __init__(self, model, log_dir="results"):
        self.model = model
        self.log_dir = log_dir

    def confusion_matrix(self, y_target, y_pred):
        y_target = y_target.astype(int)
        y_pred = np.array(y_pred).astype(int)
        TP = np.sum((y_target == 1) & (y_pred == 1))
        TN = np.sum((y_target == 0) & (y_pred == 0))
        FP = np.sum((y_target == 0) & (y_pred == 1))
        FN = np.sum((y_target == 1) & (y_pred == 0))
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

    def evaluate_metrics(self, y_target, y_pred):
        conf_matrix = self.confusion_matrix(y_target, y_pred)
        TP, TN, FP, FN = conf_matrix["TP"], conf_matrix["TN"], conf_matrix["FP"], conf_matrix["FN"]
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}

    def evaluate(self, X, y, dataset_name="test"):
        y_pred = self.model.predict(X)
        metrics = self.evaluate_metrics(y, y_pred)
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
