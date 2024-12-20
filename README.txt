## Proje Açıklaması
Bu proje, Makine Öğrenmesi (BLM5110) dersi kapsamında bir firmaya iş başvurusunda bulunan kişilerin sınav notları ile 
bir kişinin işe kabul edilip edilmeyeceğini lojistik regresyon yöntemi ile bulan makine öğrenmesi gerçekleştirilmektedir.

Projede, NumPy ile manuel olarak geliştirilen bir lojistik regresyon modeli kullanılmıştır.
Eğitim sürecindeki kayıpların görselleştirilmesi ve sınıf dağılımı grafikleri için Matplotlib kütüphanesi kullanılmıştır.
Dataset'ten veri okumak için pandas kütüphanesi kullanılmıştır.
Sınıf Dağılımı görselleştirilmesinde grafik üzerinde inceleme yapabilmek için fare imleci üzerine gelince bilgiler getiren mplcursors kullanılmıştır.

## Gereksinimler
Bu projeyi çalıştırmak için aşağıdaki Python paketleri gereklidir:
- numpy
- pandas
- matplotlib
- mplcursors

Gerekli paketleri aşağıdaki komutla yükleyebilirsiniz:
pip install -r requirements.txt

## Çalıştırma
1- Sınıf Dağılımı görselleştirilmesi:
python visualize.py

2- Model eğitimi ve loss çıktısı ve grafiği oluşturma
python train.py

3- Eğitime sokulan bir modeli metriklere  göre değerlendirme
python eval.py

### Dosya Düzeni
/InterviewResultClassifier
   |- train.py              # Eğitim ve loss çıktısı ve grafiği oluşturma
   |- eval.py               # Eğitime sokulan bir modeli metriklere  göre değerlendirme
   |- dataset.py            # Veriyi yükme ve eğitim, doğrulama, test setlerine bölme
   |- logistic_model.py     # Lojistik regresyon modeli
   |- visualize.py          # Sınıf Dağılımı görselleştirilmesi 
   |- README.txt            # Proje Açıklaması
   |- requirements.txt      # Gereksinimler
   |- dataset/hw1Data.txt   # dataset
   |- results/              # Çıktılar learning rate ve epoch bazlı olarak ayrı ayrı kaydedilir