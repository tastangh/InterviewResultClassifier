## Proje Açıklaması
Bu proje, Makine Öğrenmesi (BLM5110) dersi kapsamında bir firmaya iş başvurusunda bulunan kişilere yapılan sınav sonuçları ile
sınav notları verilen bir kişinin işe kabul edilip edilmeyeceğini lojistik regresyon yöntemi ile bulan makine öğrenmesi gerçekleştirilmektedir.

Projede, NumPy ile manuel olarak geliştirilen bir lojistik regresyon modeli kullanılmıştır.
Eğitim sürecindeki kayıpların görselleştirilmesi ve sınıf dağılımı grafikleri için Matplotlib kütüphanesi kullanılmıştır.


## Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki Python paketleri gereklidir:
- numpy
- pandas
- matplotlib

Gerekli paketleri aşağıdaki komutla yükleyebilirsiniz:
pip install -r requirements.txt

## Çalıştırma
python main.py
ve bir işlem seçin:
1: Modeli Eğit (train)
2: Modeli Test Et (eval)
3: Veriyi Görselleştir (visualize)

### Dosya Düzeni
/InterviewResultClassifier
   |- train.py
   |- eval.py
   |- dataset.py
   |- logistic_model.py
   |- visualize.py
   |- metrics.py
   |- README.txt
   |- requirements.txt
   |- dataset/hw1Data.txt
   |- results/