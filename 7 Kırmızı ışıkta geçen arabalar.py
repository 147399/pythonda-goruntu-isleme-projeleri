import cv2

# Çizgi koordinatlarını tanımla 
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

# yolo ile
import cv2
import numpy as np

# Video dosyasının yolu
video_path = 'C:/users/ASAF/Desktop/vid-and-pic/traffic_video_original.mp4'

# YOLO model dosyalarının yolları
weights_path = 'C:/users/ASAF/Desktop/yolo/yolov4.weights'
config_path = 'C:/users/ASAF/Desktop/yolo/yolov4.cfg'
classes_path = 'C:/users/ASAF/Desktop/yolo/coco.names'

# YOLO sınıf isimlerini yükle
with open(classes_path, 'r') as f:
    classes = f.read().strip().split('\n')

# YOLO modelini yükle
net = cv2.dnn.readNet(weights_path, config_path)

# Çizgi koordinatları
line_start = (700, 900)  # Çizginin başlangıç noktası
line_end = (1800, 950)  # Çizginin bitiş noktası

# Araçların geçiş sayısını tutacak sayaç ve geçiş yapan araçlar listesi
car_count = 0
crossed_cars = []

# Video kaynağını aç
video_capture = cv2.VideoCapture(video_path)

# Video döngüsü
while video_capture.isOpened():
    ret, frame = video_capture.read()

    if not ret:
        break

    # Çerçevenin boyutlarını al
    height, width = frame.shape[:2]

    # YOLO için blob oluştur
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # YOLO ağından çıkış katmanlarını alın
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Çizgiyi çizin
    cv2.line(frame, line_start, line_end, (0, 255, 0), 2)

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Sadece araç sınıflarını kontrol et (ör. araba, kamyon)
            if confidence > 0.5 and classes[class_id] in ['car', 'truck', 'bus']:
                box = detection[:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Araç dikdörtgeni
                cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 0, 255), 2)

                # Çizgiyi geçen araçları kontrol et
                if line_start[0] < center_x < line_end[0] and line_start[1] - 10 < center_y < line_start[1] + 10:
                    if (center_x, center_y) not in crossed_cars:
                        car_count += 1
                        crossed_cars.append((center_x, center_y))

    # Çizgiye geçen araç sayısını göster
    cv2.putText(frame, f'Cars Passed: {car_count}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Videoyu göster
    cv2.imshow('Traffic Monitoring', frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
video_capture.release()
cv2.destroyAllWindows()
