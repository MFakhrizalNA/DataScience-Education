import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('random_forest_model_1.joblib')
scaler = joblib.load('scaler_1.joblib')

# Fitur yang digunakan
selected_features = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade'
]

# Halaman utama
st.set_page_config(page_title="Prediksi Dropout", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")

# Form input
with st.form("input_form"):
    st.write("Masukkan informasi mahasiswa:")
    
    c1, c2 = st.columns(2)
    with c1:
        sem1_passed = st.number_input("🧾 Lulus Semester 1", min_value=0, step=1)
        sem2_passed = st.number_input("🧾 Lulus Semester 2", min_value=0, step=1)
    with c2:
        sem1_grade = st.number_input("📊 Nilai Rata-rata Semester 1", min_value=0.0, max_value=20.0, step=0.1)
        sem2_grade = st.number_input("📊 Nilai Rata-rata Semester 2", min_value=0.0, max_value=20.0, step=0.1)

    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi
if submitted:
    input_data = pd.DataFrame([{
        'Curricular_units_1st_sem_approved': sem1_passed,
        'Curricular_units_1st_sem_grade': sem1_grade,
        'Curricular_units_2nd_sem_approved': sem2_passed,
        'Curricular_units_2nd_sem_grade': sem2_grade
    }])[selected_features]

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "🎓 **Lulus (Graduate)**" if prediction == 1 else "⚠️ **Dropout**"

    st.markdown("## Hasil Prediksi:")
    st.success(f"Status Mahasiswa: {result}")