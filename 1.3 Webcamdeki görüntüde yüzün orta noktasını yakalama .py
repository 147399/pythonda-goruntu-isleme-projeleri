import cv2

# Video akışını başlatın
cap = cv2.VideoCapture(0)

# Yüz tespiti için Haar Cascade sınıflandırıcısını yükleyin
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    # Video akışından bir kare alın
    ret, frame = cap.read()

    # Eğer kare alınamadıysa döngüyü kırın
    if not ret:
        print("Kare alınamadı. Video akışını kontrol edin.")
        break

    # Renkli resmi gri tonlamaya dönüştürün
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gri tonlamalı resimde yüzleri tespit edin
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Her bir tespit edilen yüz için işlemleri yapın
    for (x, y, w, h) in faces:
        # Yüzün etrafına dikdörtgen çizin
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Yüzün ortasını hesaplayın ve işaretleyin
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    # Sonuçları gösterin
    cv2.imshow("Face Detection", frame)

    # 'q' tuşuna basılırsa döngüyü kırın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video akışını serbest bırakın ve tüm pencereleri kapatın
cap.release()
cv2.destroyAllWindows()
