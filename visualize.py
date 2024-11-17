import matplotlib.pyplot as plt
import os
import mplcursors
from dataset import DataProcessor


class DataVisualizer:
    def __init__(self, file_path, save_dir="results"):
        """
        DataVisualizer sınıfını başlatır.

        Args:
        file_path (str): Veri dosyasının yolu.
        save_dir (str): Grafiklerin kaydedileceği dizin.
        """
        self.file_path = file_path
        self.save_dir = save_dir
        self.dataset = DataProcessor(file_path)
        os.makedirs(save_dir, exist_ok=True)

    def plot_all_data(self):
        """
        Tüm veri setinin dağılımını çiz.
        """
        X, y = self.dataset.X, self.dataset.y
        plt.figure(figsize=(8, 6))
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        scatter_0 = plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
        scatter_1 = plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)
        plt.xlabel("1. Sınav Notu")
        plt.ylabel("2. Sınav Notu")
        plt.title("Tüm Veri Setinin Dağılımı")
        plt.legend()
        cursor = mplcursors.cursor([scatter_0, scatter_1], hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"1. Sınav: {sel.target[0]:.2f}\n2. Sınav: {sel.target[1]:.2f}\nSonuç: {'Kabul' if sel.artist == scatter_1 else 'Ret'}"))
        save_path = os.path.join(self.save_dir, "tüm_veri_seti_dağılımı.png")
        plt.savefig(save_path)
        plt.show()

    def plot_individual(self, X, y, title, save_file):
        """
        Bireysel setin grafiğini çizer (eğitim, doğrulama veya test seti).

        Args:
        X (ndarray): Özellikler.
        y (ndarray): Etiketler.
        title (str): Grafik başlığı.
        save_file (str): Kaydedilecek dosya adı.
        """
        plt.figure(figsize=(8, 6))
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        scatter_0 = plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
        scatter_1 = plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)
        plt.xlabel("1. Sınav Notu")
        plt.ylabel("2. Sınav Notu")
        plt.title(title)
        plt.legend()
        cursor = mplcursors.cursor([scatter_0, scatter_1], hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"1. Sınav: {sel.target[0]:.2f}\n2. Sınav: {sel.target[1]:.2f}\nSonuç: {'Kabul' if sel.artist == scatter_1 else 'Ret'}"))
        save_path = os.path.join(self.save_dir, save_file)
        plt.savefig(save_path)
        plt.show()

    def plot_splits(self):
        """
        Eğitim, doğrulama ve test setlerinin dağılımlarını ayrı ayrı çiz.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.dataset.split_data()
        if X_train is not None and y_train is not None:
            self.plot_individual(X_train, y_train, "Eğitim Seti", "eğitim_seti_dağılımı.png")
        if X_val is not None and y_val is not None:
            self.plot_individual(X_val, y_val, "Doğrulama Seti", "doğrulama_seti_dağılımı.png")
        if X_test is not None and y_test is not None:
            self.plot_individual(X_test, y_test, "Test Seti", "test_seti_dağılımı.png")


if __name__ == "__main__":
    data_path = "dataset/hw1Data.txt"

    # Görselleştirme sınıfı oluştur
    visualizer = DataVisualizer(data_path)

    # Tüm veri setini çiz
    visualizer.plot_all_data()

    # Eğitim, doğrulama ve test setlerini çiz
    visualizer.plot_splits()
