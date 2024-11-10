import numpy as np
from logistic_model import LogisticRegressionSGD
from dataset import load_data, split_data
import os
from datetime import datetime

def confusion_matrix(y_target, y_pred):
    """
    Confusion matrix hesaplar ve TP, TN, FP, FN değerlerini döner.
    
    Args:
        y_target (ndarray): Gerçek etiketler
        y_pred (ndarray): Tahmin edilen etiketler
        
    Returns:
        dict: Confusion matrix {TP, TN, FP, FN}
    """
    # Veri türlerini eşitle
    y_target = y_target.astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    TP = np.sum((y_target == 1) & (y_pred == 1))
    TN = np.sum((y_target == 0) & (y_pred == 0))
    FP = np.sum((y_target == 0) & (y_pred == 1))
    FN = np.sum((y_target == 1) & (y_pred == 0))
    
    # Confusion matrix değerlerini ekrana yazdırarak kontrol edin
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def evaluate_metrics(y_target, y_pred):
    """
    Confusion matrix ile accuracy, precision, recall ve F1-score hesaplar.
    
    Args:
        y_target (ndarray): Gerçek etiketler
        y_pred (ndarray): Tahmin edilen etiketler
        
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    conf_matrix = confusion_matrix(y_target, y_pred)
    TP = conf_matrix["TP"]
    TN = conf_matrix["TN"]
    FP = conf_matrix["FP"]
    FN = conf_matrix["FN"]
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.0
    
    # Precision ve Recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    
    # Her metrik sonucunu ekrana yazdırarak kontrol edin
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")
    
    return accuracy, precision, recall, f1_score


def evaluate_model_on_test_set(model, X_test, y_test):
    """
    Modeli test seti üzerinde değerlendirir ve performans metriklerini döndürür.
    
    Args:
        model -- eğitimli lojistik regresyon modeli (LogisticRegressionSGD)
        X_test -- test seti özellik verileri (ndarray)
        y_test -- test seti hedef verileri (ndarray)
        
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    # Test seti üzerinde tahmin yap
    y_test_pred = model.predict(X_test)
    
    # Test setindeki gerçek etiketleri ve tahminleri yazdır
    print("Gerçek Test Etiketleri:", y_test)
    print("Modelin Tahminleri:", y_test_pred)
    
    # Performans metriklerini hesapla
    return evaluate_metrics(y_test, y_test_pred)


def save_evaluation_results(accuracy, precision, recall, f1_score, learning_rate, epochs, log_dir="results"):
    """
    Test seti üzerinde elde edilen performans metriklerini learning rate ve epoch bilgisi ile .txt dosyasına kaydeder.
    
    Args:
        accuracy -- doğruluk oranı (float)
        precision -- precision değeri (float)
        recall -- recall değeri (float)
        f1_score -- F1-score değeri (float)
        learning_rate -- modelin öğrenme oranı (float)
        epochs -- toplam epoch sayısı (int)
        log_dir -- sonuçların kaydedileceği dizin yolu (str)
    """
    # Dosya adında learning rate ve epoch bilgisi ekle
    log_path = os.path.join(log_dir, f"eval_results_lr{learning_rate}_epochs{epochs}.txt")

    # Dizin yoksa oluştur
    os.makedirs(log_dir, exist_ok=True)

    # Sonuçları dosyaya yaz
    with open(log_path, "w") as f:
        f.write("Test Sonuçları:\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
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

    # Model parametreleri
    learning_rate = 0.001
    epochs = 5000

    # Modeli eğit
    model = LogisticRegressionSGD(learning_rate=learning_rate, epochs=epochs)
    model.fit(X_train, y_train, X_val, y_val)

    # Test seti üzerinde değerlendirme yap ve sonuçları kaydet
    accuracy, precision, recall, f1_score = evaluate_model_on_test_set(model, X_test, y_test)
    save_evaluation_results(accuracy, precision, recall, f1_score, learning_rate, epochs)

    # Sonuçları konsola yazdır
    print("\n=== Test Seti Sonuçları ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

if __name__ == "__main__":
    eval_model()
