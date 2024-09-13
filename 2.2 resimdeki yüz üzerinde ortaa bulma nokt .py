import cv2

# Resim dosyasının yolunu belirleyin
image_path = "C:/Users/ASAF/Desktop/vid-and-pic/images.jpeg"

# Resmi yükleyin
img = cv2.imread(image_path)

# Haar Cascade XML dosyasını yükleyin (yüz tespiti için)
face_cascade = cv2.CascadeClassifier("C:/Users/ASAF/Desktop/Classifiers/haarcascade_frontalface_default.xml")

# Renkli resmi gri tonlamaya dönüştürün
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı resimde yüzleri tespit edin
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dosyanın düzgün yüklendiğini kontrol edin
if img is None:
    print(f"Resim dosyası açılamadı. Dosya yolu veya dosya adı: {image_path}")
else:
    # Her bir tespit edilen yüz için işlemleri yapın
    for (x, y, w, h) in faces:
        # Yüzün ortasının x koordinatını hesaplayın
        center_x = x + w // 2
        # Yüzün ortasının y koordinatını hesaplayın
        center_y = y + h // 2

        # Yüzün etrafına dikdörtgen çizin
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Yüzün ortasına yeşil bir nokta çizin
        cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), -1)

    # Sonuçları gösterin (yüzlerin etrafında dikdörtgenler ve ortasında noktalar ile)
    cv2.imshow("Middle of Face", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
