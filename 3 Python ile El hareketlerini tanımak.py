import cv2
import mediapipe
import pyttsx3

# Kameradan görüntü yakalamak için VideoCapture nesnesi oluşturuluyor
camera = cv2.VideoCapture(0)

# Sesli geri bildirim için pyttsx3 motoru başlatılıyor
engine = pyttsx3.init()

# Mediapipe Hands modülü için gerekli olan nesneler oluşturuluyor
mpHands = mediapipe.solutions.hands

# Hands sınıfı ile el hareketlerini tanımlamak için bir nesne oluşturuluyor
hands = mpHands.Hands()

# El hareketlerinin çizimi için yardımcı fonksiyonlar
mpdraw = mediapipe.solutions.drawing_utils

# El başparmağının yukarı olup olmadığını kontrol etmek için bir bayrak değişkeni
checkThumpsUp = False

while True:

   # Kameradan bir görüntü yakalanıyor
   success, img = camera.read()

   # OpenCV ile alınan BGR formatındaki görüntü, RGB formatına dönüştürülüyor
   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   # Hands modülü ile el hareketleri işleniyor
   hlms = hands.process(imgRGB)

   # Görüntünün boyutlarını alıyoruz (yükseklik, genişlik, kanal sayısı)
   height, width, channel = img.shape

   # Eğer el hareketi algılanmışsa
   if hlms.multi_hand_landmarks:
      # Algılanan her bir el için işlem yapılacak
      for handlandmarks in hlms.multi_hand_landmarks:
         # El hareket noktaları görüntü üzerinde çiziliyor
         mpdraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)
         print(handlandmarks.landmark)  # Her bir parmak noktalarının koordinatları yazdırılıyor

         # Her bir parmak noktası (landmark) için döngüye giriliyor
         for fingerNum, landmark in enumerate(handlandmarks.landmark):
            # Parmak noktalarının x ve y pozisyonları, görüntünün genişliği ve yüksekliği ile ölçekleniyor
            positionX, positionY = int(landmark.x * width), int(landmark.y * height)

            # El noktalarının pozisyonları görüntüye yazılabilir (yorum satırı haline getirilmiş)
            # cv2.putText(img, str(fingerNum),(positionX,positionY),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            # Başparmak yukarıda mı diye kontrol ediliyor
            if fingerNum > 4 and landmark.y < handlandmarks.landmark[2].y:
               break  # Eğer diğer parmaklar yukarıda ise, döngü kırılıyor

            # Son parmak noktası (serçe parmağı) yukarıdaysa ve başparmak yukarıdaysa, bayrak değişkeni True yapılıyor
            if fingerNum == 20 and landmark.y > handlandmarks.landmark[2].y:
               checkThumpsUp = True

   # Eğer başparmak yukarıdaysa
   if checkThumpsUp:
      # Pyttsx3 motoru ile "Thumps Up" ifadesi sesli olarak söyleniyor
      engine.say("Thumps Up")
      engine.runAndWait()
      break  # Döngü sonlandırılıyor

   # Kameradan alınan görüntü ekranda gösteriliyor
   cv2.imshow("camera : ", img)
   
   # 'q' tuşuna basılırsa döngü sonlandırılıyor
   if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Kamera serbest bırakılıyor ve tüm pencereler kapatılıyor
camera.release()
cv2.destroyAllWindows()
#ANP