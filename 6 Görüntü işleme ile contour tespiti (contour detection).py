import cv2
import numpy as np 


# Resmi yükleyin. Dosya yolu belirtilmemiş, bu nedenle geçerli bir dosya yolu ile doldurulmalıdır.
img = cv2.imread("C:/Users/ASAF/Desktop/Datasets/sekiller.jpg")   
#img2 = img.copy()  # Orijinal resmi kopyalayarak img2 oluşturun

# Dosyanın düzgün yüklendiğini kontrol et
if img is None:
    print("PNG dosyası açılamadı. Dosya yolunu kontrol edin.")
else:
    # Resmi göster
    cv2.imshow("PNG Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Resmi gri tonlamaya dönüştürün
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gri tonlamalı resmi binary (ikili) resme dönüştürün
ret, imggray = cv2.threshold(gray, 150, 250, cv2.THRESH_BINARY)

# Contours (şekil sınırları) ve hiyerarşiyi bulun
x = 0 
contours, hiearray = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Her bir konturu işleyin
for cnt in contours:
    print(len(contours))  # Bulunan kontur sayısını yazdırın
    # Eğer konturun hiyerarşisinde üst kontur yoksa
    if hiearray[0][0][3] == -1:
        x = x + 1
        if x == 2:  # İkinci konturu işleyin
            cv2.drawContours(img, [cnt], -1, (255, 0, 0), 30)  # Konturu resme çizin
            print(cnt)  # Konturu yazdırın
            pass

# Gri tonlamalı resmi cv2.COLOR_BGR2GRAY formatında tekrar dönüştürün 
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  

# Köşe tespiti yapın
corners = cv2.goodFeaturesToTrack(gray, 5, 0.01, 10)  # Gri tonlamalı resimde köşeleri tespit edin
corners = np.int0(corners)  # Köşe koordinatlarını tam sayılara dönüştürün

# Her bir köşeyi işleyin ve resme çizin
a = 1
for i in corners:
    x, y = i.ravel()  # Köşe koordinatlarını düzleştirin
    cv2.circle(img, (x, y), 3, (0, 0, 255), 20)  # Köşe noktasını resme çizin
    a = a + 1

# Sonuçları görüntüleyin
cv2.imshow("img", img)  # Kontur ve köşe işaretlemeleri içeren resmi gösterin
#cv2.imshow("img2", img2)  

# Pencereyi kapatmak için herhangi bir tuşa basılmasını bekleyin
cv2.waitKey(0)
cv2.destroyAllWindows() 
