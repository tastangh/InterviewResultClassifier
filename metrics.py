import numpy as np

def confusion_matrix(y_target, y_pred):
    """
    Confusion matrix hesaplar ve TP, TN, FP, FN değerlerini döner.
    
    Args:
        y_target (ndarray): Gerçek etiketler
        y_pred (ndarray): Tahmin edilen etiketler
        
    Returns:
        dict: Confusion matrix {TP, TN, FP, FN}
    """
    TP = np.sum((y_target == 1) & (y_pred == 1))
    TN = np.sum((y_target == 0) & (y_pred == 0))
    FP = np.sum((y_target == 0) & (y_pred == 1))
    FN = np.sum((y_target == 1) & (y_pred == 0))
    
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
    print(str(TP) + " " + str(TN) + " " + str(FP) + " " + str(FN))
    
    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    # Precision ve Recall
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    
    # F1-Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0
    
    return accuracy, precision, recall, f1_score
