# main.py
from train import Trainer
from eval import Evaluator
from visualize import plot_data
from dataset import DataProcessor

def main():
    # Veri dosyasının yolu
    data_path = "dataset/hw1Data.txt"
    
    # Sınıf dağılım grafiği oluştur
    # plot_data(data_path)
    
    # Model eğitimi
    learning_rate=0.00001
    epochs=15000
    trainer = Trainer(learning_rate=learning_rate, epochs=epochs)
    model, training_losses, validation_losses = trainer.train(data_path)
    
    # Model değerlendirme
    dataset = DataProcessor(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()
    
    evaluator = Evaluator(model)
    train_metrics = evaluator.evaluate(X_train, y_train, "Eğitim Seti")
    val_metrics = evaluator.evaluate(X_val, y_val, "Doğrulama Seti")
    test_metrics = evaluator.evaluate(X_test, y_test, "Test Seti")
    
    # Değerlendirme sonuçlarını kaydet
    evaluator.save_results(train_metrics, val_metrics, test_metrics, learning_rate=learning_rate, epochs=epochs)
    
    print("Eğitim ve değerlendirme tamamlandı. Sonuçlar 'results' klasöründe kaydedildi.")

if __name__ == "__main__":
    main()
