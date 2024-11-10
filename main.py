from train import train_model
from eval import eval_model
from visualize import plot_data

# TODO : . Eğitim, doğrulama ve test örnekleri için accuracy, precision, recall ve f-scrore hesabını yapınız.(Şuan sadece test için yapıyoz)

def main():
    """
    Eğitim, test veya görselleştirme işlemlerini başlatır.
    Kullanıcıdan bir işlem seçmesini ister ve ilgili işlemi çağırır.
    """
    print("Lütfen bir işlem seçin:")
    print("1: Modeli Eğit (train)")
    print("2: Modeli Test Et (eval)")
    print("3: Veriyi Görselleştir (visualize)")
    
    choice = input("Seçiminizi yapın (1/2/3): ")

    if choice == '1':
        print("\nModel eğitiliyor...\n")
        train_model()
        print("\nEğitim tamamlandı.")
        
    elif choice == '2':
        print("\nModel test ediliyor...\n")
        eval_model()
        print("\nTest işlemi tamamlandı.")

    elif choice == '3':
        print("\nVeri görselleştiriliyor...\n")
        plot_data("dataset/hw1Data.txt")
        print("\nGörselleştirme tamamlandı.")

    else:
        print("Geçersiz seçim. Lütfen 1, 2 veya 3 girin.")

if __name__ == "__main__":
    main()
