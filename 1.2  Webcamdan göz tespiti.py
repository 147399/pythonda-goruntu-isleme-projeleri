import numpy as np
import cv2

# Yüz ve göz tespiti için gerekli olan Haar Cascade sınıflandırıcılarını yüklüyoruz.
face_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/haarcascade_eye.xml")

# Bilgisayarın varsayılan kamerasından video akışını başlatıyoruz.
cap = cv2.VideoCapture(0)

# Sonsuz bir döngü ile sürekli olarak kameradan görüntü alıyoruz.
while True: 
    # Kameradan bir kare okuyoruz, 'ret' okuma işleminin başarılı olup olmadığını döner, 'img' ise görüntü verisini tutar.
    ret, img = cap.read()

    # Görüntüyü gri tonlamaya çeviriyoruz, çünkü yüz ve göz tespiti gri tonlamalı görüntülerde daha hızlı çalışır.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gri tonlamalı görüntüde yüz tespiti yapıyoruz. 1.3 ölçeklendirme faktörü ve en az 5 komşu dikdörtgen kullanılır.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Bulunan her yüz için bir dikdörtgen çiziyoruz.
    for (x, y, w, h) in faces:
        # Yüzün etrafına mavi bir dikdörtgen çiziliyor.
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

        # Yüz bölgesini (ROI - Region of Interest) gri tonlamalı ve renkli olarak alıyoruz.
        roi_gray = gray[y:y+h, x:x+w]  # Yüz bölgesinin gri tonlamalı kısmı.
        roi_color = img[y:y+h, x:x+w]  # Yüz bölgesinin renkli kısmı.

        # Yüz bölgesi içinde göz tespiti yapıyoruz.
        eyes = eyes_cascade.detectMultiScale(roi_gray)
        i = 0 
        # Bulunan her göz için sarı bir dikdörtgen çiziyoruz.
        for (ex, ey, ew, eh) in eyes:
            i +=1 
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 1)
            if ( i ==2 ):
                break
    # İşlenmiş görüntüyü OpenCV penceresinde gösteriyoruz.
    cv2.imshow("title", img)

    # Eğer 'q' tuşuna basılırsa döngüyü sonlandırıyoruz.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Video yakalama işlemini serbest bırakıyoruz.
cap.release()

# Açık olan tüm OpenCV pencerelerini kapatıyoruz.
cv2.destroyAllWindows()
