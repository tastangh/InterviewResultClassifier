import matplotlib.pyplot as plt
import os
import mplcursors
from dataset import DataProcessor

def plot_data(file_path, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    """
    Verideki örneklerin iki sınıfa dağılımını görmek için grafikleri oluşturur.
    1. Tüm veri setini çizer.
    2. Eğitim, doğrulama ve test setlerini ayrı ayrı çizer.

    Argümanlar:
    file_path -- veri dosyasının yolu
    X_train, y_train -- Eğitim seti
    X_val, y_val -- Doğrulama seti
    X_test, y_test -- Test seti
    """
    # DataProcessor sınıfı ile veriyi yükle
    dataset = DataProcessor(file_path)
    X, y = dataset.X, dataset.y

    # Klasör oluşturma
    save_path = "results/"
    os.makedirs(save_path, exist_ok=True)

    # Tüm veri setini çiz
    plt.figure(figsize=(8, 6))
    class_0 = X[y == 0]
    class_1 = X[y == 1]
    scatter_0 = plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
    scatter_1 = plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)
    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.title("Tüm Veri Setinin Dağılımı")
    plt.legend()
    # mplcursors ile etkileşimli bilgi
    cursor = mplcursors.cursor([scatter_0, scatter_1], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"1. Sınav: {sel.target[0]:.2f}\n2. Sınav: {sel.target[1]:.2f}\nSonuç: {'Kabul' if sel.artist == scatter_1 else 'Ret'}"))
    plt.savefig(os.path.join(save_path, "tüm_veri_seti_dağılımı.png"))
    plt.show()

    # Eğitim, doğrulama ve test setlerini ayrı ayrı çiz
    if X_train is not None and y_train is not None:
        plot_individual(X_train, y_train, "Eğitim Seti", os.path.join(save_path, "eğitim_seti_dağılımı.png"))
    if X_val is not None and y_val is not None:
        plot_individual(X_val, y_val, "Doğrulama Seti", os.path.join(save_path, "doğrulama_seti_dağılımı.png"))
    if X_test is not None and y_test is not None:
        plot_individual(X_test, y_test, "Test Seti", os.path.join(save_path, "test_seti_dağılımı.png"))

def plot_individual(X, y, title, save_path):
    """
    Bireysel setin grafiğini çizer (eğitim, doğrulama veya test seti).

    Argümanlar:
    X, y -- Özellikler ve etiketler
    title -- Grafik başlığı
    save_path -- Kaydedilecek dosya yolu
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
    # mplcursors ile etkileşimli bilgi
    cursor = mplcursors.cursor([scatter_0, scatter_1], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"1. Sınav: {sel.target[0]:.2f}\n2. Sınav: {sel.target[1]:.2f}\nSonuç: {'Kabul' if sel.artist == scatter_1 else 'Ret'}"))
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    data_path = "dataset/hw1Data.txt"

    # DataProcessor sınıfını kullanarak veri bölme
    dataset = DataProcessor(data_path)
    X_train, y_train, X_val, y_val, X_test, y_test = dataset.split_data()

    # plot_data fonksiyonunu çağırma
    plot_data(data_path, X_train, y_train, X_val, y_val, X_test, y_test)
