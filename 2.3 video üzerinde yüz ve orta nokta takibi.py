import cv2
import cv2.data

# Video dosyasının yolunu belirtir ve video dosyasını yükler
video_path = 'C:/users/ASAF/Desktop/vid-and-pic/wt.mp4'  
video_capture = cv2.VideoCapture(video_path)

# Göz ve yüz tespiti için Haarcascades sınıflandırıcılarını yükler
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Videoyu oynatmaya başlar
while video_capture.isOpened():
    # Her kareyi video dosyasından oku
    ret, frame = video_capture.read()

    # Eğer video sona erdiyse veya kare okunamazsa döngüden çık
    if not ret:
        print("video acılamadı")
        break

    # Kareyi gri tonlamalı hale getirir (Yüz ve göz algılama gri tonlama üzerinde daha iyi çalışır)
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit eder (detectMultiScale ile yüz bölgelerini bulur)
    faces = face_cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    # Tespit edilen her yüz için işlem yapar
    for (x, y, w, h) in faces:
        # Yüzün etrafına bir dikdörtgen çizer
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 85, 0), 5)

        # Yüzün merkez noktasını hesaplarc      
        center_x = x + w // 2
        center_y = y + h // 2

        # Yüzün merkez noktasına küçük bir daire çizer
        cv2.circle(frame, (center_x, center_y), 5, (85, 85, 0), -1)

        # Yüz bölgesini ayıkla
        roi_gray = gray_scale[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]

        # Yüz içinde gözleri tespit et
        eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(2, 2))

        # Tespit edilen her göz için işlem yapar
        for (ex, ey, ew, eh) in eyes:
            # Gözlerin etrafına bir dikdörtgen çizer
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

    # Kareyi gösterir (Video penceresinde gösterilen anlık kare)
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında videoyu durdurur
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırakır ve tüm OpenCV pencerelerini kapatır
video_capture.release()
cv2.destroyAllWindows()


"""
# Video dosyasının yolunu belirtir ve video dosyasını yükler
video_path = 'C:/users/ASAF/Desktop/wt.mp4'  
video_capture = cv2.VideoCapture(video_path)

# Göz ve yüz tespiti için Haarcascades sınıflandırıcılarını yükler
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray_scale = cv2.cvtColor(video_capture , cv2.COLOR_BGR2GRAY)

# Videoyu oynatmaya başlar
while video_capture.isOpened():
    # Her kareyi video dosyasından okur
    ret, frame = video_capture.read()

    # Eğer video sona erdiyse veya kare okunamazsa döngüden çıkar
    if not ret:
        break

    # Kareyi gri tonlamalı hale getirir (Yüz ve göz algılama gri tonlama üzerinde daha iyi çalışır)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüzleri tespit eder (detectMultiScale ile yüz bölgelerini bulur)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    
    # Tespit edilen her yüz için işlem yapar
    for (x, y, w, h) in faces:
        # Yüzün etrafına bir dikdörtgen çizer
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 85, 0), 5)

        # Yüzün merkez noktasını hesaplar
        center_x = x + w // 2
        center_y = y + h // 2


        # Yüzün merkez noktasına küçük bir daire çizer
        cv2.circle(frame, (center_x, center_y), 5, (85, 85, 0), -1)
    roi_gray = gray_scale[y:y+h , x:x+w]
    roi_color = video_capture[y:y+h , x:x+w]
    # Gözleri tespit eder
    eyes = eyes_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(2, 2))
    eyes1 = eyes_cascade.detectMultiScale(roi_gray)
    # Tespit edilen her göz için işlem yapar
    for (ex, ey, ew, eh) in eyes:
        # Gözlerin etrafına bir dikdörtgen çizer
        cv2.rectangle(roi_color, (x+ex, y+ey), (ex + ew, ey + eh), (0, 0, 255), 1)

    # Kareyi gösterir (Video penceresinde gösterilen anlık kare)
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında videoyu durdurur
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Video dosyasını serbest bırakır ve tüm OpenCV pencerelerini kapatır
video_capture.release()
cv2.destroyAllWindows()


"""