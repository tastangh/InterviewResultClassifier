import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Veriyi dosya yolundaki dosyadan alır. Virgüllerle ayrılmış datayı sütunlara ayırır.
    İlk iki sütün olan 1.sınav notu ve 2.sınav notunu özellik olarak X'e atar.
    3.sütun olan Kabul/Ret çıktısını y'ye atar

    Argümanlar:
    file_path -- veri dosyasının yolu (str)
    
    Returns:
    X -- Özellik verileri
    y -- Çıktı verileri
    """
    # Değerler virgülle ayrılmıştır. O yüzden delimiter'i virgül yapıyoruz.
    data = pd.read_csv(file_path, header=None, delimiter=',').values
    X = data[:, :2]   #İlk iki sütün olan 1.sınav notu ve 2.sınav notunu özellik olarak X'e atar.
    y = data[:, 2]    #3.sütun olan Kabul/Ret çıktısını y'ye atar
    return X, y

def split_data(X, y, train_ratio=0.6, val_ratio=0.2):
    """
    Veride verilen örneklerin ilk %60’ını eğitim, sonraki %20’sini doğrulama, kalan %20’sini test için böler.
    
    Argümanlar:
    X -- özellik verileri 
    y -- hedef çıktı verileri 
    train_ratio -- eğitim seti oranı (varsayılan 0.6) 
    val_ratio -- doğrulama seti oranı (varsayılan 0.2) 
    
    Returns:
    X_train, y_train -- eğitim seti verileri 
    X_val, y_val -- doğrulama seti verileri 
    X_test, y_test -- test seti verileri 
    """
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test
