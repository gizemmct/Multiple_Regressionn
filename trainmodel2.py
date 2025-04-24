import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Veri kümesini yükle
file_path = 'Laptop_price.csv'
data = pd.read_csv(file_path)

# Kategorik verileri sayısal hale getir
label_encoder = LabelEncoder()
data['Brand'] = label_encoder.fit_transform(data['Brand'])

# Özellikleri ve hedefi ayır
X = data[['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']]
y = data['Price']

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Modeli kaydet
joblib.dump(model, 'laptop_price_rating_model.pkl')

print("Model başarıyla eğitildi ve kaydedildi.")


