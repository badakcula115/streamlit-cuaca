import streamlit as st
import joblib

# Muat model
model = joblib.load('model.pkl')

# Judul Aplikasi
st.title("Prediksi Cuaca dengan Machine Learning")

# Input dari Pengguna
suhu = st.number_input("Suhu Rata-rata", value=25.0)
kelembaban = st.number_input("Kelembaban Relatif Rata-rata", value=70.0)
curah_hujan = st.number_input("Curah Hujan", value=5.0)
kecepatan_angin = st.number_input("Kecepatan Angin (m/s)", value=3.0)

# Tombol Prediksi
if st.button("Prediksi"):
    try:
        prediction = model.predict([[suhu, kelembaban, curah_hujan, kecepatan_angin]])[0]
        hasil = "Hujan" if prediction == 1 else "Tidak Hujan"
        st.success(f"Hasil Prediksi: {hasil}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
