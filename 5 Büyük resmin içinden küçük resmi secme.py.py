import cv2

# Büyük resim ve küçük resim dosyalarını yükleyin
picture = cv2.imread("C:/Users/ASAF/Desktop/vid-and-pic/araba.jpeg")
smallpicture = cv2.imread("C:/Users/ASAF/Desktop/vid-and-pic/tekerlek2.jpg")

# Dosyanın düzgün yüklendiğini kontrol edin
if picture is None:
    print("Büyük resim dosyası açılamadı. Dosya yolunu kontrol edin.")
if smallpicture is None:
    print("Küçük resim dosyası açılamadı. Dosya yolunu kontrol edin.")

# Eğer her iki dosya da başarıyla yüklendiyse, işlemleri gerçekleştirin
if picture is not None and smallpicture is not None:
    # Resimleri gri tonlamalıya dönüştürün
    picture_gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    smallpicture_gray = cv2.cvtColor(smallpicture, cv2.COLOR_BGR2GRAY)

    # Şablon eşleştirme uygulayın
    find = cv2.matchTemplate(picture_gray, smallpicture_gray, cv2.TM_CCOEFF_NORMED)
    
    # Eşleşmenin minimum ve maksimum değerlerini ve konumlarını bulun
    minval, maxval, minloc, maxloc = cv2.minMaxLoc(find)
    topleft = maxloc  # En yüksek eşleşme puanına sahip konum

    # Küçük resmin boyutlarını alın ve büyük resim üzerinde dikdörtgen çizin
    h, w = smallpicture_gray.shape[:2]
    bottomright = (topleft[0] + w, topleft[1] + h)
    cv2.rectangle(picture, topleft, bottomright, (0, 255, 0), 1)  # Dikdörtgeni yeşil renkte çizin

    # Sonuçları görselleştirin
    cv2.imshow("find picture", picture)  # Büyük resmi ve bulunan eşleşmeyi göster
    cv2.imshow("find", find)  # Eşleştirme sonuçlarını göster
    cv2.imshow("smallpicture", smallpicture_gray)  # Küçük resmi göster

    # Pencereyi kapatmak için herhangi bir tuşa basılmasını bekleyin
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
