import numpy as np
import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        """
        DataProcessor sınıfı, dosya yolundaki veriyi yükler ve eğitim, doğrulama, test setlerine böler.

        Args:
            file_path -- veri dosyasının yolu
        """
        self.file_path = file_path
        self.X, self.y = self.load_data()

    def load_data(self):
        """
        Veriyi dosya yolundaki dosyadan alır ve özellik (X) ve etiket (y) olarak ayırır.
        
        Returns:
            X -- Özellik verileri
            y -- Çıktı verileri
        """
        data = pd.read_csv(self.file_path, header=None, delimiter=',').values
        X = data[:, :2]
        y = data[:, 2]
        return X, y

    def split_data(self, train_ratio=0.6, val_ratio=0.2):
        """
        Veriyi eğitim, doğrulama ve test setleri olarak böler.
        
        Args:
            train_ratio -- Eğitim seti oranı
            val_ratio -- Doğrulama seti oranı
        
        Returns:
            X_train, y_train -- Eğitim seti verileri
            X_val, y_val -- Doğrulama seti verileri
            X_test, y_test -- Test seti verileri
        """
        train_size = int(len(self.X) * train_ratio)
        val_size = int(len(self.X) * val_ratio)

        X_train, y_train = self.X[:train_size], self.y[:train_size]
        X_val, y_val = self.X[train_size:train_size + val_size], self.y[train_size:train_size + val_size]
        X_test, y_test = self.X[train_size + val_size:], self.y[train_size + val_size:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
