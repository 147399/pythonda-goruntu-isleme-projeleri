import pandas as pd
import numpy as np 
import os 
from sklearn.model_selection import train_test_split 
import cv2

# Veri setinin kaydedileceği dizini tanımlıyoruz.
DATA_DIR  = "./data"

# DATA_DIR dizini mevcut değilse, oluşturuluyor.
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Toplamda kaç sınıf olduğunu ve her sınıf için kaç tane görüntü toplanacağını belirliyoruz.
number_of_class= 3  # Üç farklı sınıf (kategori) için veri toplanacak.
data_size = 100  # Her sınıf için 100 görüntü toplanacak.

# Web kamerası üzerinden video akışını başlatıyoruz.
cap = cv2.VideoCapture(0)

# Her sınıf için veri toplama işlemi
for j in range(number_of_class):    
    # Her sınıf için ayrı bir klasör oluşturuluyor.
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print("data class {}", format(j))  # Hangi sınıf için veri toplandığını belirten mesajı yazdırıyoruz.

    # Kullanıcıdan görüntüleri kaydetmek için hazır olmasını bekliyoruz.
    while True:
        _, frame = cap.read()  # Kameradan bir kare okunuyor.
        # Kullanıcıya talimat veren bir metin görüntünün üzerine ekleniyor.
        cv2.putText(frame, "hazır olunca q bass ", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

        cv2.imshow("frame", frame)  # Görüntü ekrana yansıtılıyor.
        if cv2.waitKey(25) & 0xFF == ord("q"):  # Kullanıcı 'q' tuşuna bastığında döngü sonlanıyor.
            break
    
    counter = 0  # Görüntü sayacı sıfırlanıyor.

    # Belirtilen sayıda görüntü toplama işlemi
    while counter < data_size:
        _, frame = cap.read()  # Kameradan bir kare okunuyor.
        cv2.imshow("frame", frame)  # Görüntü ekrana yansıtılıyor.
        cv2.waitKey(25)  # 25 milisaniye bekleniyor.
        # Görüntü belirtilen sınıf klasörüne kaydediliyor.
        cv2.imwrite(os.path.join(DATA_DIR, str(j), "{}.jpg".format(counter)), frame)

        counter += 1  # Görüntü sayacı artırılıyor.

# Video akışı sonlandırılıyor ve tüm pencereler kapatılıyor.
cap.release()
cv2.destroyAllWindows()