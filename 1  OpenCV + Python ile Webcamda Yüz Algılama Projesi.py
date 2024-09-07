import numpy as np
import cv2

# Kameradan görüntü yakalamak için VideoCapture nesnesi oluşturuluyor
vid = cv2.VideoCapture(0)

# OpenCV'nin Haar Cascade yüz algılama modelini yüklemek için CascadeClassifier kullanılıyor
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while(True):
    # Kameradan bir kare yakalanıyor
    ret, frame = vid.read()

    # Görüntü gri tonlamaya çevriliyor çünkü Haar Cascade daha iyi performans gösterir
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Yüzler, gri tonlamalı görüntü üzerinde algılanıyor
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)

    # Algılanan her bir yüz için bir dikdörtgen çiziliyor
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (85, 255, 0), 3)

    # İşlenmiş görüntü ekranda gösteriliyor
    cv2.imshow("title", frame)
    
    # 'q' tuşuna basılırsa döngü sonlandırılıyor
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
# Kamera serbest bırakılıyor ve tüm pencereler kapatılıyor
vid.release()
cv2.destroyAllWindows()
