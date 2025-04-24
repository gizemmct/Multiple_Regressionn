# arayuz2.py

import streamlit as st
import joblib
import numpy as np

# Modeli yükle
model = joblib.load("laptop_price_rating_model.pkl")

st.title("Laptop Fiyat Tahmin Uygulaması")

# Kullanıcıdan giriş verilerini al
brand = st.selectbox("Marka Seçiniz", ["Acer", "Asus", "Dell", "HP", "Lenovo"])
processor_speed = st.slider("İşlemci Hızı (GHz)", 1.0, 5.0, 2.5)
ram_size = st.slider("RAM (GB)", 2, 64, 8)
storage_capacity = st.slider("Depolama Kapasitesi (GB)", 128, 2000, 512)
screen_size = st.slider("Ekran Boyutu (inç)", 10.0, 18.0, 14.0)
weight = st.slider("Ağırlık (kg)", 1.0, 5.0, 2.5)

# Kullanıcının girdiği marka değerini encode etmek için aynı sıralama
brand_list = ["Acer", "Asus", "Dell", "HP", "Lenovo"]
brand_encoded = brand_list.index(brand)

# Tahmin butonu
if st.button("TAHMİN"):
    input_data = np.array([[brand_encoded, processor_speed, ram_size, storage_capacity, screen_size, weight]])
    prediction = model.predict(input_data)
    st.success(f"Tahmini Fiyat: {prediction[0]:,.2f} TL")
