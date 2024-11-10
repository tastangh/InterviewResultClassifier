from logistic_model import LogisticRegressionSGD
from dataset import load_data, split_data
from metrics import evaluate_metrics

def evaluate_model_on_test_set(model, X_test, y_test):
    """
    Modeli test seti üzerinde değerlendirir ve performans metriklerini döndürür.
    
    Argümanlar:
    model -- eğitimli lojistik regresyon modeli (LogisticRegressionSGD)
    X_test -- test seti özellik verileri (ndarray)
    y_test -- test seti hedef verileri (ndarray)
    
    Returns:
    accuracy, precision, recall, f1_score -- değerlendirme metrikleri (float)
    """
    y_test_pred = model.predict(X_test)
    print("Tahminler:", y_test_pred)  # Tahmin sonuçlarını yazdırarak kontrol edin
    return evaluate_metrics(y_test, y_test_pred)

import os
from datetime import datetime

def save_evaluation_results(accuracy, precision, recall, f1_score, log_dir="results"):
    """
    Test seti üzerinde elde edilen performans metriklerini zaman damgalı bir .txt dosyasına kaydeder.
    
    Argümanlar:
    accuracy -- doğruluk oranı (float)
    precision -- precision değeri (float)
    recall -- recall değeri (float)
    f1_score -- F1-score değeri (float)
    log_dir -- sonuçların kaydedileceği dizin yolu (str)
    """
    # Zaman damgası eklemek
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_results_{timestamp}.txt")

    # Dizin yoksa oluştur
    os.makedirs(log_dir, exist_ok=True)

    # Sonuçları dosyaya yaz
    with open(log_path, "w") as f:
        f.write("Test Sonuçları:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1_score:.4f}\n")


def eval_model():
    """
    Test verisi üzerinde modeli değerlendirir ve sonuçları konsolda gösterir ve bir dosyaya kaydeder.
    """
    # Veriyi yükle ve böl
    X, y = load_data("dataset/hw1Data.txt")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # Modeli eğit
    model = LogisticRegressionSGD(learning_rate=0.01, epochs=100)
    model.fit(X_train, y_train, X_val, y_val)

    # Test seti üzerinde değerlendirme yap ve sonuçları kaydet
    accuracy, precision, recall, f1_score = evaluate_model_on_test_set(model, X_test, y_test)
    save_evaluation_results(accuracy, precision, recall, f1_score)

    # Sonuçları konsola yazdır
    print("\n=== Test Seti Sonuçları ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    eval_model()
