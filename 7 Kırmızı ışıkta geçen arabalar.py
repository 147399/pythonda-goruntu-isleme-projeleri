import cv2

# Çizgi koordinatlarını tanımla (örneğin, (x1, y1) ve (x2, y2) noktaları)
line_start = (700, 900)  # Çizginin başlangıç noktası
line_end = (1800, 950)    # Çizginin bitiş noktası

# Araç tespiti için Haar Cascade sınıflandırıcısını yükle
video_path = 'C:/users/ASAF/Desktop/vid-and-pic/traffic_video_original.mp4'  
video_capture = cv2.VideoCapture(video_path)

# Araç tespiti için Haar Cascade sınıflandırıcısını yükler
cascade_path = "C:/users/ASAF/Desktop/classifiers/cars.xml"
car_cascade = cv2.CascadeClassifier(cascade_path)

# Araçların geçiş sayısını saymak için bir sayaç oluştur
car_count = 0
crossed_cars = []  # Geçiş yapan araçların merkez koordinatlarını saklamak için

# Videoyu oynatmaya başla
while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', 640, 480)  # 640x480 boyutunda pencere

    # Kareyi gri tonlamalı hale getir
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Araçları tespit et
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Çizgiyi video üzerine çiz
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    for (x, y, w, h) in cars:
        car_center = (x + w // 2, y + h // 2)

        # Araç merkezinin çizgiyi geçip geçmediğini kontrol et
        if line_start[0] < car_center[0] < line_end[0] and line_start[1] - 10 < car_center[1] < line_start[1] + 10:
            # Araç geçişini say
            if car_center not in crossed_cars:
                car_count += 1
                crossed_cars.append(car_center)

        # Araç etrafına dikdörtgen çizer
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Çizgiye geçen araç sayısını göster
    cv2.putText(frame, f'Cars Passed: {car_count}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Kareyi göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında videoyu durdurur
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Video kaynağını serbest bırak ve pencereleri kapat
video_capture.release()
cv2.destroyAllWindows()
