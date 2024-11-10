import numpy as np

# Sigmoid aktivasyon fonksiyonu
def sigmoid(z):
    """
    Sigmoid aktivasyon fonksiyonu, giriş değerini [0, 1] aralığında bir olasılığa dönüştürür.
    
    Argümanlar:
    z -- girdi değeri veya değerler dizisi (float veya ndarray)
    
    Returns:
    result -- sigmoid fonksiyonunun uygulandığı değer veya değerler dizisi (float veya ndarray)
    """
    return 1 / (1 + np.exp(-z))

# Cross-Entropy Loss fonksiyonu
def cross_entropy_loss(y_true, y_pred):
    """
    İkili sınıflandırma için cross-entropy loss hesaplar.
    
    Argümanlar:
    y_true -- gerçek hedef değeri (0 veya 1) (int veya float)
    y_pred -- tahmin edilen olasılık değeri (0 ile 1 arasında) (float)
    
    Returns:
    loss -- cross-entropy kayıp değeri (float)
    """
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Logistic Regression Modeli
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=100):
        """
        Lojistik regresyon modelini başlatır.
        
        Argümanlar:
        learning_rate -- öğrenme oranı (varsayılan 0.01) (float)
        epochs -- eğitim tekrar sayısı (varsayılan 100) (int)
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0
    
    def fit(self, X, y, X_val=None, y_val=None):
        """
        Lojistik regresyon modelini verilen eğitim verisiyle eğitir.
        
        Argümanlar:
        X -- eğitim özellik verileri (ndarray)
        y -- eğitim hedef verileri (ndarray)
        X_val -- doğrulama özellik verileri (varsayılan None) (ndarray)
        y_val -- doğrulama hedef verileri (varsayılan None) (ndarray)
        
        Returns:
        training_losses -- her epoch sonunda eğitim kayıplarını içeren liste (list)
        validation_losses -- her epoch sonunda doğrulama kayıplarını içeren liste (list)
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
                y_pred = sigmoid(linear_model).squeeze()
                loss = cross_entropy_loss(yi, y_pred)
                epoch_loss += loss
                
                # Ağırlık güncellemeleri
                dW = (y_pred - yi) * xi
                dB = y_pred - yi
                self.weights -= self.learning_rate * dW.squeeze()
                self.bias -= self.learning_rate * dB
            
            # Eğitim kaybını kaydet
            training_losses.append(epoch_loss / m)
            
            # Doğrulama kaybını hesapla (Eğer doğrulama seti varsa)
            if X_val is not None and y_val is not None:
                val_loss = 0
                for j in range(len(X_val)):
                    val_pred = sigmoid(np.dot(X_val[j], self.weights) + self.bias)
                    val_loss += cross_entropy_loss(y_val[j], val_pred)
                validation_losses.append(val_loss / len(X_val))
        
        return training_losses, validation_losses
    
    def predict(self, X):
        """
        Eğitimli lojistik regresyon modelini kullanarak verilen veri seti için tahmin yapar.
        
        Argümanlar:
        X -- tahmin yapılacak özellik verileri (ndarray)
        
        Returns:
        predictions -- tahmin edilen sınıf etiketleri (0 veya 1) listesi (list)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in predictions]
