import numpy as np

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Lojistik regresyon modelini başlatır.
        
        Args:
            learning_rate -- öğrenme oranı
            epochs -- epoch sayısı
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def _sigmoid(self, z):
        """
        Sigmoid aktivasyon fonksiyonu.
        
        Args:
            z -- girdi değeri veya değerler dizisi
        
        Returns:
            result -- sigmoid fonksiyonunun sonucu
        """
        return 1 / (1 + np.exp(-z))

    def _cross_entropy_loss(self, y_target, y_pred):
        """
        İkili sınıflandırma için cross-entropy loss hesaplar.
        
        Args:
            y_target -- gerçek hedef değeri
            y_pred -- tahmin edilen olasılık değeri
        
        Returns:
            loss -- cross-entropy kayıp değeri
        """
        return - (y_target * np.log(y_pred) + (1 - y_target) * np.log(1 - y_pred))

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Lojistik regresyon modelini eğitir.
        
        Args:
            X -- eğitim verisi
            y -- eğitim etiketleri
            X_val -- doğrulama verisi
            y_val -- doğrulama etiketleri
        
        Returns:
            training_losses -- eğitim kayıpları listesi
            validation_losses -- doğrulama kayıpları listesi
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
                
                # Tahmin ve loss hesaplama
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
            X -- tahmin yapılacak veri seti
        
        Returns:
            predictions -- tahmin edilen etiketler listesi
        """
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]
