import numpy as np

class LogisticRegressionModel:
    """
    Basit bir lojistik regresyon modeli sınıfı.
    Model, verilen öğrenme oranı ve epoch sayısına göre eğitim yapar.
    
    Özellikler:
        learning_rate (float): Modeli eğitmek için kullanılan öğrenme oranı.
        epochs (int): Modelin eğitimde geçireceği epoch sayısı.
        weights (numpy.ndarray): Modelin özellik ağırlıkları.
        bias (float): Modelin bias değeri.
    """

    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Lojistik regresyon modelini başlatır.
        
        Args:
            learning_rate (float): Öğrenme oranı. Varsayılan 0.01.
            epochs (int): Epoch sayısı. Varsayılan 100.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def _sigmoid(self, z):
        """
        Sigmoid aktivasyon fonksiyonu.
        
        Args:
            z (numpy.ndarray): Girdi değeri veya değerler dizisi.
        
        Returns:
            numpy.ndarray: Sigmoid fonksiyonunun sonucu.
        """
        return 1 / (1 + np.exp(-z))

    def _cross_entropy_loss(self, y_target, y_pred, epsilon=1e-10):
        """
        İkili sınıflandırma için cross-entropy kaybını hesaplar.
        
        Args:
            y_target (int): Gerçek hedef değeri (0 veya 1).
            y_pred (float): Tahmin edilen olasılık değeri.
            epsilon (float): Sayısal kararlılık için kullanılan küçük sabit, varsayılan 1e-10.
        
        Returns:
            float: Cross-entropy kayıp değeri.
        """
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return - (y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Lojistik regresyon modelini eğitim verileri ile eğitir.
        
        Args:
            X (numpy.ndarray): Eğitim verisi, her satır bir örneği temsil eder.
            y (numpy.ndarray): Eğitim etiketleri, 0 veya 1 değerlerinden oluşur.
            X_val (numpy.ndarray, opsiyonel): Doğrulama verisi, varsayılan None.
            y_val (numpy.ndarray, opsiyonel): Doğrulama etiketleri, varsayılan None.
        
        Returns:
            training_losses (list of float): Her epoch için eğitim kayıpları listesi.
            validation_losses (list of float): Her epoch için doğrulama kayıpları listesi (eğer doğrulama verisi varsa).
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        training_losses = []
        validation_losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            for i in range(m):
                xi = X[i].reshape(1, -1)
                yi = y[i]
                
                # Tahmin ve kayıp hesaplama
                linear_model = np.dot(xi, self.weights) + self.bias
                y_pred = self._sigmoid(linear_model).squeeze()
                loss = self._cross_entropy_loss(yi, y_pred)
                epoch_loss += loss

                # Ağırlık güncellemeleri
                dW = (y_pred - yi) * xi
                dB = y_pred - yi
                self.weights -= self.learning_rate * dW.squeeze()
                self.bias -= self.learning_rate * dB
            
            training_losses.append(epoch_loss / m)
            
            # Doğrulama kaybını hesapla (Eğer doğrulama seti varsa)
            if X_val is not None and y_val is not None:
                val_loss = 0
                for j in range(len(X_val)):
                    val_pred = self._sigmoid(np.dot(X_val[j], self.weights) + self.bias)
                    val_loss += self._cross_entropy_loss(y_val[j], val_pred)
                validation_losses.append(val_loss / len(X_val))

        return training_losses, validation_losses

    def predict(self, X):
        """
        Eğitimli model ile tahmin yapar.
        
        Args:
            X (numpy.ndarray): Tahmin yapılacak veri seti, her satır bir örneği temsil eder.
        
        Returns:
            list of int: Tahmin edilen etiketler (0 veya 1) listesi.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]
