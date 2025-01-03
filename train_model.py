import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Data cuaca contoh
data = {
    'Suhu_Rata_rata': [32, 25, 29, 31, 28, 35, 22, 30, 27, 33],
    'Kelembaban_Relatif_Rata_rata': [85, 70, 75, 80, 65, 90, 60, 72, 78, 88],
    'Curah_Hujan': [10, 0, 5, 12, 0, 20, 0, 8, 15, 18],
    'Kecepatan_Angin_dalam_meter_per_detik': [6, 3, 4, 7, 2, 9, 3, 5, 8, 10],
    'Hujan': [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  # 1 untuk Hujan, 0 untuk Tidak Hujan
}

# Mengonversi ke DataFrame
df = pd.DataFrame(data)

# Memisahkan fitur (X) dan target (y)
X = df.drop('Hujan', axis=1)
y = df['Hujan']

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Menyimpan model ke file
joblib.dump(model, 'model.pkl')

print("Model berhasil dibuat dan disimpan sebagai 'model.pkl'")
