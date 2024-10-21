import  pandas as pd 
import os 
import pickle  
import mediapipe as mp 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import numpy as np

# Pickle dosyasını açıp veri ve etiketleri yükler.
data_dict = pickle.load(open("data.pickle", "rb"))

# Veriyi ve etiketleri numpy dizilerine dönüştürür.
data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

print(data.shape)  # data'nın boyutunu verir
print(labels.shape)  # labels'ın boyutunu verir

x = data
y = labels
# Veriyi eğitim ve test setlerine böler.
x_train, x_test, y_train, y_test = train_test_split(x ,y , train_size=0.70, shuffle=True)

# Rastgele Orman modelini oluşturur ve eğitir.
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Test seti üzerinde tahmin yapar ve doğruluğu hesaplar.
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

# Başarı oranını yazdırır.
print("Başarı:", score * 100)

# Modeli pickle dosyasına kaydeder.
f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()