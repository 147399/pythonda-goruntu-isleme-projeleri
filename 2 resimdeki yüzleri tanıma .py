import cv2
import matplotlib.pyplot as plt 
import numpy as np 

# Resmi yükle
img =  cv2.imread("C:/Users/ASAF/Desktop/Datasets/images.jpeg")

# Resmi gri tonlamaya çevir
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı resmi matplotlib ile göster
plt.imshow(gray_scale) 
plt.show()

# Haar Cascade sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/haarcascade_frontalface_default.xml")

# Yüzleri tespit et
faces = face_cascade.detectMultiScale(gray_scale, 1.2, 2)

# Tespit edilen yüzlerin etrafına dikdörtgen çiz
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

# Yüz tespiti yapılan resmi göster
cv2.imshow("face detection", img)
cv2.waitKey(0)  # Pencereyi kapatmak için bir tuşa basılmasını bekle
cv2.destroyAllWindows() 

# Resimden yüz tespiti yapan bir fonksiyon
def detecFromImage(image):
    # Haar Cascade sınıflandırıcıyı yükle
    face_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/haarcascade_frontalface_default.xml")

    # Verilen resmi oku
    img = cv2.imread(image)

    # Resmi gri tonlamaya çevir
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray_scale, 1.2, 1)

    # Tespit edilen yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Tespit edilen yüzleri içeren resmi göster
    cv2.imshow("insanlar", img)
    cv2.waitKey()  
    cv2.destroyAllWindows()  



#MTCNN ile 

import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# MTCNN yüz tespiti modeli yüklenir
detector = MTCNN()

# Görüntüyü yükle
image_path = 'C:/Users/ASAF/Desktop/vid-and-pic/insanlar.jpeg'  # Kendi resim yolunuzu kullanın
image = cv2.imread(image_path)

# BGR'yi RGB'ye dönüştür (cv2 görüntüyü BGR olarak yükler)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Yüzleri tespit et
faces = detector.detect_faces(image_rgb)

# Her tespit edilen yüz için işlemleri yap
for face in faces:
    # Yüzün dikdörtgen koordinatlarını al
    x, y, width, height = face['box']
    
    # Yüzü kare içine al (kırmızı renkte bir dikdörtgen çiz)
    cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (255, 0, 0), 2)

    # Yüz üzerindeki landmark noktalarını çizer (gözler, burun, ağız)
    for key, point in face['keypoints'].items():
        cv2.circle(image_rgb, point, 5, (0, 255, 0), -2)

# Sonucu görüntüle
plt.imshow(image_rgb)
plt.axis('off')  # Eksenleri kapat
plt.show()
