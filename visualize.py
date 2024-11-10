import matplotlib.pyplot as plt
import os
from dataset import load_data

def plot_data(file_path):
    """
    Verideki örneklerin iki sınıfa dağılımını görmek için x eksenini 1. sınav notu, y eksenini 2. sınav
    notu için kullanarak ve iki sınıfa ait örnekleri iki farklı renk ve farklı şekilde göstererek
    örnekleri çizdirir ve png olarak kaydeder.

    Argümanlar:
    file_path -- veri dosyasının yolu
    """
    # Veriyi yükle
    X, y = load_data(file_path)

    # Sınıfları ayır
    class_0 = X[y == 0]
    class_1 = X[y == 1]

    # Grafik oluştur
    plt.figure(figsize=(8, 6))
    plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
    plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)

    # Grafik etiketleri
    plt.xlabel("1. Sınav Notu")
    plt.ylabel("2. Sınav Notu")
    plt.title("İki Sınıfın Sınav Notlarına Göre Dağılımı")
    plt.legend()

    # Klasör oluşturma ve grafiği kaydetme
    save_path = "results/graphs"
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, "sınıf_dağılımı.png"))

    # Grafiği göster
    plt.show()
