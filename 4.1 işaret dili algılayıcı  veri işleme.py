import pickle 
import os
import mediapipe as mp
import cv2

# Mediapipe bileşenlerini yükleme
mp_hands = mp.solutions.hands  # El tespiti için Hands modülünü kullanıyoruz.
mp_drawing = mp.solutions.drawing_utils  # Elde edilen el noktalarını çizmek için yardımcı fonksiyonlar.
mp_drawing_styles = mp.solutions.drawing_styles  # Çizim stilini belirlemek için kullanılır.

# Hands sınıfını başlatma
# static_image_mode=True: Statik görüntülerde el tespiti yapacağımızı belirtiyor.
# min_detection_confidence=0.3: Tespit için minimum güven seviyesi.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Veri dizini
DATA_DIR  = "./data"

# Boş listeler oluşturma
data = []  # El hareketlerinden elde edilen özellik vektörlerini saklamak için.
labels = []  # Bu özellik vektörlerine karşılık gelen sınıf etiketlerini saklamak için.

# Veri dizinindeki her alt klasör için döngü
for dir_ in os.listdir(DATA_DIR):  # DATA_DIR içindeki alt klasörleri listeler (her biri bir sınıfı temsil eder).
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):  # Her alt klasördeki görüntüleri listeler.
        data_aux = []  # Her görüntü için özellikleri saklamak üzere geçici bir liste.
        x_ = []  
        y_ = []

        # Görüntüyü yükleme
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Görüntüyü yükler.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV BGR formatında çalışır, bu nedenle RGB'ye dönüştürülür.

        # El tespiti
        result = hands.process(img_rgb)  # Görüntü üzerinde el tespiti yapar.

        # Eğer elde tespit edilmişse, her bir landmark (el noktası) üzerinde döngü
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:  # Birden fazla el tespit edilmişse, her biri için işlem yapılır.
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Noktanın x koordinatını alır.
                    y = hand_landmarks.landmark[i].y  # Noktanın y koordinatını alır.

                    x_.append(x)  # x koordinatlarını listeye ekler.
                    y_.append(y)  # y koordinatlarını listeye ekler.

                # El hareketlerini normalize etme
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    # x ve y koordinatlarını minimum değerden çıkararak normalize eder.
                    data_aux.append(x - min(x_))  
                    data_aux.append(y - min(y_))
                
                # Bu görüntüye ait verileri ve sınıf etiketini listelere ekleme
                data.append(data_aux)  # Özellikleri 'data' listesine ekler.
                labels.append(dir_)  # Sınıf etiketini 'labels' listesine ekler.

# Veriyi pickle dosyasına kaydetme
f = open("data.pickle", "wb")  # 'wb' modunda (yazma ve ikili mod) dosya açılır.
pickle.dump({"data": data, "labels": labels}, f)  # Veriler ve etiketler 'data.pickle' dosyasına kaydedilir.
f.close()  # Dosya kapatılır.
