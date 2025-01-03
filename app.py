from flask import Flask, render_template, request
import joblib

# Inisialisasi Flask app
app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Ambil data dari form
            suhu = float(request.form["Suhu_Rata_rata"])
            kelembaban = float(request.form["Kelembaban_Relatif_Rata_rata"])
            curah_hujan = float(request.form["Curah_Hujan"])
            kecepatan_angin = float(request.form["Kecepatan_Angin_dalam_meter_per_detik"])
            
            # Lakukan prediksi
            prediction = model.predict([[suhu, kelembaban, curah_hujan, kecepatan_angin]])[0]
            prediction = "Hujan" if prediction == 1 else "Tidak Hujan"
        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
