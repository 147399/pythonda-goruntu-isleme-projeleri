import cv2 
import matplotlib.pyplot as plt
import numpy as np


img =  "C:/Users/ASAF/Desktop/vid-and-pic/yuz.jpeg"
# Yüz ve göz tespiti yapmak için bir fonksiyon tanımlıyoruz.
def detecFaces_EyesFromImage(image):

    # Haar Cascade Classifier'larını yükleyerek yüz ve göz algılama için modelleri hazır hale getiriyoruz.
    face_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/classifiers/haarcascade_frontalface_default.xml")
    eyes_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/classifiers/haarcascade_eye.xml")

    # Verilen görüntü dosyasını yüklüyoruz.
    img = cv2.imread(image)

    # Görüntüyü gri tonlamaya çeviriyoruz, çünkü yüz ve göz algılamada gri tonlamalı görüntü kullanılır.
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit ediyoruz. detectMultiScale fonksiyonu, görüntüdeki yüzlerin koordinatlarını döndürür.
    faces = face_cascade.detectMultiScale(gray_scale, 1.2, 1)

    # Algılanan yüzlerin her biri için işlem yapıyoruz.
    for (x, y, w, h) in faces:
        # Yüzün etrafına bir yeşil dikdörtgen çiziyoruz.
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Yüz bölgesini gri tonlamalı görüntüden alıyoruz.
        roi_gray = gray_scale[y:y+h, x:x+w]

        # Yüz bölgesini renkli görüntüden alıyoruz.
        roi_color = img[y:y+h, x:x+w]

        # Yüz içinde gözleri tespit ediyoruz.
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        
        
        # Algılanan gözler için işlem yapıyoruz.
        for (ex, ey, ew, eh) in eyes:     
            # Gözlerin etrafına bir kırmızı dikdörtgen çiziyoruz.
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 0)

    # Algılanan yüzleri ve gözleri gösteren bir pencere açıyoruz.
    cv2.imshow("faces and eyes detected", img)

    # Pencerenin kapanması için bir tuşa basılmasını bekliyoruz.
    cv2.waitKey()

    # Açılan tüm pencereleri kapatıyoruz.
    cv2.destroyAllWindows()

# Verilen görüntü dosyasını fonksiyonla işliyoruz.
detecFaces_EyesFromImage("C:/Users/ASAF/Desktop/vid-and-pic/yuz.jpeg")

#ANP