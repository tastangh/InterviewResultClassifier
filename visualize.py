import matplotlib.pyplot as plt
import os
import mplcursors  # Tooltip için gerekli kütüphane
from dataset import load_data

# Veriyi yükle
X, y = load_data("dataset/hw1Data.txt")

# Sınıfları ayır
class_0 = X[y == 0]
class_1 = X[y == 1]

# Grafik oluştur
plt.figure(figsize=(8, 6))
scatter_ret = plt.scatter(class_0[:, 0], class_0[:, 1], color='red', marker='x', label='Ret (0)', alpha=0.6)
scatter_kabul = plt.scatter(class_1[:, 0], class_1[:, 1], color='green', marker='o', label='Kabul (1)', alpha=0.6)

# Grafik etiketleri
plt.xlabel("1. Sınav Notu")
plt.ylabel("2. Sınav Notu")
plt.title("İki Sınıfın Sınav Notlarına Göre Dağılımı")
plt.legend()

# Tooltip oluşturma: Noktaların üzerine gelindiğinde sınav notları görünsün
cursor = mplcursors.cursor([scatter_ret, scatter_kabul], hover=True)

# Tooltip içeriğini sınav notları ile doldurma
@cursor.connect("add")
def on_add(sel):
    index = sel.index
    if sel.artist == scatter_ret:
        sel.annotation.set_text(f"Ret (0)\n1. Sınav: {class_0[index, 0]:.2f}\n2. Sınav: {class_0[index, 1]:.2f}")
    else:
        sel.annotation.set_text(f"Kabul (1)\n1. Sınav: {class_1[index, 0]:.2f}\n2. Sınav: {class_1[index, 1]:.2f}")

# Klasör yolu
save_path = "results/graphs"
os.makedirs(save_path, exist_ok=True) 

# Grafiği PNG olarak kaydet
plt.savefig(os.path.join(save_path, "sınıf_dağılımı.png"))

# Grafiği göster
plt.show()
