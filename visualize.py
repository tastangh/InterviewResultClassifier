# visualize.py
import matplotlib.pyplot as plt
import os
import mplcursors
from dataset import DataProcessor

def plot_data(file_path):
    """
    Verideki örneklerin iki sınıfa dağılımını görmek için x eksenini 1. sınav notu, y eksenini 2. sınav
    notu için kullanarak ve iki sınıfa ait örnekleri iki farklı renk ve farklı şekilde göstererek
    örnekleri çizdirir ve png olarak kaydeder.

    Argümanlar:
    file_path -- veri dosyasının yolu
    """
    # DataProcessor sınıfı ile veriyi yükle
    dataset = DataProcessor(file_path)
    X, y = dataset.X, dataset.y

    # Sınıfları ayır
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    scatter_0 = plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
    scatter_1 = plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)

    # Grafik etiketleri
    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.title("İki Sınıfın Sınav Notlarına Göre Dağılımı")
    plt.legend()

    # Klasör oluşturma ve grafiği kaydetme
    save_path = "results/"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "sınıf_dağılımı.png"))

    # Her bir nokta üzerine fareyi getirince sınav notlarını ve kabul/red bilgisini gösterme
    cursor = mplcursors.cursor([scatter_0, scatter_1], hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"1. Sınav: {sel.target[0]:.2f}\n2. Sınav: {sel.target[1]:.2f}\nSonuç: {'Kabul' if sel.artist == scatter_1 else 'Ret'}"))

    # Grafiği göster
    plt.show()

if __name__ == "__main__":
    # Sabit dosya yolu tanımla
    data_path = "dataset/hw1Data.txt"
    
    # plot_data fonksiyonunu çağırma
    plot_data(data_path)
