# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Maju

## Business Understanding

Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus.

### Permasalahan Bisnis
Tingkat dropout mahasiswa di Jaya Jaya Institut tergolong tinggi, yang dapat:
- Penurunan reputasi institusi.
- Dampak negatif pada akreditasi dan kepercayaan publik.
- Kerugian finansial akibat hilangnya pemasukan dari mahasiswa.
- Beban tambahan pada proses akademik dan administrasi.

### Cakupan Proyek
- Memprediksi kemungkinan dropout mahasiswa menggunakan data historis akademik dan non-akademik.
- Visualisasi monitoring dropout secara real-time untuk membantu pengambilan keputusan oleh pihak kampus.

### Cakupan Fungsional Proyek
1. Pengolahan Data Mahasiswa
   - Pembersihan dan normalisasi data akademik dan keuangan.
   - Transformasi data kategorikal (misalnya gender) dan numerik (misalnya nilai) menjadi format siap model.

2. Model Prediksi Dropout
Training model menggunakan Random Forest Classifier, Fitur yang digunakan meliputi:
   - Jenis kelamin
   - Nilai rata-rata semester 1
   - Nilai rata-rata semester 2
Model disimpan dalam bentuk .joblib bersama encoder untuk keperluan inferensi otomatis.

3. Antarmuka Prediksi (Streamlit)
   - Form input interaktif untuk memasukkan data mahasiswa baru.
   - Prediksi dropout ditampilkan secara real-time dengan visual feedback.

4. Dashboard Monitoring Dropout (Looker Studio)
Visualisasi dropout berdasarkan faktor-faktor utama:
   - Nilai akademik (semester 1 & 2)
   - Jenis kelamin
Tampilan:
   - Jumlah total dropout
   - Distribusi dropout berdasarkan gender
   - Perbandingan nilai rata-rata mahasiswa dropout vs tidak dropout.

5. Penyimpanan Data di PostgreSQL
Seluruh data mahasiswa dan hasil prediksi disimpan dalam PostgreSQL. Basis data ini menjadi sumber utama untuk Looker Studio melalui koneksi terintegrasi (via Google Cloud Connector atau service account).

6. Integrasi dengan Google Looker Studio


### Persiapan
| Jenis      | Keterangan                                                                 |
|------------|------------------------------------------------------------------------------|
| Title      | Jaya Jaya Institut                                                   |
| Source     | [github](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance) |
| Visibility | Public                                                                      |

Setup environment:

### 1. Buat Conda Environment

```bash
conda create -n education python=3.10 -y
conda activate education
```

### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

Menjalankan predict dropout
### 1. Clone Repository

```bash
git clone https://github.com/MFakhrizalNA/DataScience-Education.git
cd DataScience-Education
```
### 2. Menjalankan Aplikasi Streamlit

```bash
streamlit run pred_dropout.py
```

# Data Understanding
Dataset ini berisi informasi akademik, administratif, dan demografis mahasiswa. Dimana kolom yang dijelaskan pada bagian ini adalah yeng mempunyai korelasi tinggi terhadap keputusan dropout.

| Kolom                               | Deskripsi                                                                 |
|------------------------------------|---------------------------------------------------------------------------|
| `Application_mode`                 | Jalur pendaftaran mahasiswa (numerik yang merepresentasikan mode tertentu) |
| `Gender`                           | Jenis kelamin (1 = Laki-laki, 0 = Perempuan)                             |
| `Age_at_enrollment`               | Usia mahasiswa saat mendaftar (dalam tahun)                              |
| `Debtor`                           | Status hutang pendidikan (1 = Ya, 0 = Tidak)                              |
| `Scholarship_holder`              | Status penerima beasiswa (1 = Ya, 0 = Tidak)                              |
| `Tuition_fees_up_to_date`         | Apakah pembayaran SPP tepat waktu (1 = Ya, 0 = Tidak)                     |
| `Curricular_units_1st_sem_approved` | Jumlah mata kuliah yang lulus di semester 1                             |
| `Curricular_units_1st_sem_grade`  | Nilai rata-rata semester 1                                               |
| `Curricular_units_2nd_sem_approved` | Jumlah mata kuliah yang lulus di semester 2                             |
| `Curricular_units_2nd_sem_grade`  | Nilai rata-rata semester 2                                               |
| `Status`                           | Status akhir mahasiswa (1 = Graduate, 0 = Dropout, 2=Enroll)                        |

## 1. Bivariate - EDA
Bagian ini menyajikan visualisasi hubungan antara dua variabel dalam dataset. Teknik yang digunakan antara lain:
- Crosstab untuk melihat distribusi kategorikal antar variabel.
- Scatterplot untuk mengamati pola hubungan antar fitur numerik.
- Boxplot untuk membandingkan persebaran nilai dalam kategori yang berbeda
## 2. Multivariate - EDA
Pada tahap ini, dilakukan analisis visual terhadap beberapa variabel secara bersamaan untuk menemukan pola yang kompleks. Visualisasi yang digunakan meliputi:
- Matriks Korelasi untuk mengukur kekuatan hubungan antar fitur numerik.
- Pairplot untuk menggambarkan distribusi sepasang variabel sekaligus.
## 3. Pengecekan Missing Values, Duplicated dan Outliers
- Dataset tidak mengandung missing value (nilai kosong).
- Tidak ditemukan baris duplikat dalam data.
- Terdapat outlier, yaitu nilai-nilai ekstrem yang menyimpang dari pola data umum, yang divisualisasikan menggunakan boxplot.
# Data Preparation

## 1. Mengatasi Outliers
Outlier adalah data yang berada jauh dari distribusi mayoritas. Keberadaannya dapat mengganggu pelatihan model, terutama pada algoritma yang peka terhadap distribusi. Dalam proyek ini, pendekatan yang digunakan adalah Interquartile Range (IQR), yaitu metode statistik yang mengidentifikasi data ekstrem di luar rentang kuartil untuk mengurangi pengaruh nilai-nilai anomali.

## 2. Encoding Data Kategorikal
Beberapa fitur, seperti `Status`, bersifat kategorikal dan tidak bisa langsung digunakan dalam model pembelajaran mesin. Oleh karena itu, dilakukan proses encoding, yaitu mengubah nilai kategorikal menjadi bentuk numerik agar dapat diterima oleh algoritma.

## 3. Seleksi Fitur (Feature Selection)
Feature selection dilakukan untuk memilih fitur-fitur yang paling berpengaruh terhadap target. Dengan menghilangkan variabel yang tidak relevan, proses ini membantu meningkatkan akurasi model, mengurangi risiko overfitting, serta mempercepat waktu pelatihan. Dalam proyek ini, fitur dipilih berdasarkan tingkat korelasinya terhadap variabel target.

Fitur terpilih: ['Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade']

## 4. Splitting Data
Pemisahan dataset menjadi data latih dan data uji merupakan langkah penting dalam pembangunan model machine learning. Tujuannya adalah untuk mengevaluasi performa model pada data yang belum pernah digunakan saat pelatihan, sehingga hasil evaluasi mencerminkan kemampuan model untuk melakukan generalisasi terhadap data baru. Pada model ini pembagiannya menjadi 20% data testing dan 80% data training.

## 4. Standarisasi
Standardisasi dilakukan untuk menyamakan skala variabel numerik dalam dataset, dengan menyamakan skala, model lebih cepat konvergen dan performanya meningkat.

## 6. SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE (Synthetic Minority Over-sampling Technique) adalah metode yang digunakan untuk menangani imbalance atau ketidakseimbangan jumlah data antar kelas dalam dataset. Teknik ini bekerja dengan menciptakan sampel sintetis dari kelas minoritas melalui interpolasi data yang ada. Hasilnya, model menjadi lebih mampu mengenali pola dari kelas yang sebelumnya terpinggirkan, sehingga performa klasifikasi menjadi lebih seimbang.


# Model Development
## Random Forest Classifier (RFC)
Random Forest Classifier (RFC) adalah bagian dari algoritma ensemble learning yang digunakan untuk klasifikasi data. Algoritma ini bekerja dengan membangun banyak pohon keputusan (decision trees) dan menggabungkan hasilnya (voting) untuk menentukan kelas akhir.

### Cara Kerja Random Forest
- Membuat banyak decision tree dari subset acak data dan fitur.
- Setiap pohon membuat prediksi sendiri.
- RFC mengambil hasil mayoritas dari seluruh pohon (voting) sebagai hasil akhir klasifikasi.

Dalam proyek ini, RFC digunakan untuk memprediksi apakah seorang mahasiswa akan dropout atau tidak berdasarkan data akademik dan demografis.

### Kelebihan Random Forest
- Lebih tahan terhadap overfitting dibanding satu pohon keputusan.
- Dapat menangani data kategorikal dan numerik tanpa perlu normalisasi rumit.
- Cocok untuk prediksi dropout, karena bisa menangkap pola kompleks antara berbagai faktor.

----
# Evaluation
## Clasification Report
**Classification Report** adalah ringkasan metrik evaluasi untuk model klasifikasi. Laporan ini memberikan wawasan tentang kinerja model dengan membandingkan label yang diprediksi terhadap label sebenarnya.
Classification Report biasanya mencakup metrik berikut untuk setiap kelas:

| **Metrik**  | **Deskripsi**                                                                                   |
|-------------|--------------------------------------------------------------------------------------------------|
| **Precision** | Proporsi prediksi positif yang benar-benar benar (True Positive / (True Positive + False Positive)) |
| **Recall**    | Proporsi data positif yang berhasil diprediksi dengan benar (True Positive / (True Positive + False Negative)) |
| **F1-score**  | Rata-rata harmonis dari precision dan recall, berguna saat ingin seimbangkan keduanya          |
| **Support**   | Jumlah kemunculan aktual dari masing-masing kelas dalam data                                   |

**Hasil Classification SVC** pada model ini adalah sebagai berikut:

                        precision    recall  f1-score   support

               0          0.82      0.78      0.80       274
               1          0.86      0.89      0.88       423

         accuracy                             0.85       697
         macro avg        0.84      0.84      0.84       697
      weighted avg        0.85      0.85      0.85       697

## Business Dashboard
![dashboard](./fakhrizal25-dashboard.png)

Untuk mempermudah pemantauan dan analisis  performa siswa secara berkala, telah dibuat sebuah dashboard interaktif menggunakan looker studio: [dashboar edukasi](https://lookerstudio.google.com/reporting/9c2cb1f5-a794-419c-87cd-0f5fcb6da818). Dashboard ini menyajikan visualisasi data yang intuitif dan informatif mengenai faktor-faktor yang mempengaruhi performa siswa, seperti:

- Distribusi perfroma siswa berdasarkan Gender
- Faktor mempengaruhi dropout seperti nilai semester 1 dan 2, kelulusan mata kuliah semester 1 dan 2.

## Solusi Machine Learning
Solusi machine learning ini dikembangkan sebagai alat bantu awal bagi Jaya Jaya Institut untuk memprediksi kemungkinan dropout mahasiswa berdasarkan data historis akademik dan administratif.

Aplikasi dibangun menggunakan Streamlit dan berfungsi sebagai dashboard interaktif sederhana yang memiliki dua fitur utama:

    - Prediksi Status Mahasiswa (Dropout / Graduate)

    - Pengguna dapat memasukkan data individual mahasiswa melalui form input.

    - Model prediksi menggunakan algoritma Random Forest Classifier dengan parameter `random_state = 42`.
    
    - Output yang ditampilkan berupa status: Dropout atau Graduate.

🎯 Akses aplikasi online: [pred_app](https://datascience-education-bacjgu73fsmlppmu5n5cav.streamlit.app/)

## Conclusion
Melalui analisis dan implementasi model prediksi dropout menggunakan algoritma Random Forest Classifier (RFC), proyek ini berhasil mengidentifikasi pola-pola penting yang memengaruhi kemungkinan seorang mahasiswa tidak menyelesaikan studinya. kesimpulan utama:
1. Nilai akademik semester 1 dan 2 sangat berpengaruh terhadap risiko dropout:
   - Mahasiswa dengan rata-rata nilai rendah di semester 1 dan 2 memiliki kemungkinan lebih besar untuk dropout.
   - Terdapat korelasi kuat antara performa awal dan keberlanjutan studi.
2. Faktor jenis kelamin (gender) juga menunjukkan kontribusi terhadap prediksi dropout:
   - Dalam beberapa analisis, terdapat perbedaan kecenderungan antara kelompok mahasiswa berdasarkan gender, meskipun kontribusinya tidak sebesar nilai akademik.
3. Model Random Forest memberikan hasil yang stabil dan interpretatif, memungkinkan tim akademik untuk menilai kontribusi masing-masing fitur (feature importance).

Dengan integrasi ke Looker Studio dan data dari PostgreSQL, Jaya Jaya Institut kini dapat:
1. Melakukan pemantauan real-time terhadap mahasiswa yang berisiko tinggi dropout.
2. Menyajikan dashboard interaktif dengan metrik seperti:
   - Distribusi rata-rata nilai per semester berdasarkan status dropout.
   - Komparasi performa akademik antara mahasiswa dropout dan non-dropout.
   - Segmentasi risiko berdasarkan gender.
3. Menyusun kebijakan berbasis data, bukan asumsi.


### Rekomendasi Action Items
Beberapa rekomendasi item aksi yang dapat diterapkan oleh Jaya-jaya Institut, adalah sebagai berikut:
1. Fokus pada Performa Akademik Awal
   - Lakukan monitoring ketat terhadap nilai semester pertama dan kedua.
   - Mahasiswa dengan rata-rata nilai <10 (misalnya) dapat dikategorikan sebagai kelompok berisiko.
   - Terapkan sistem alert berbasis nilai untuk tindakan lebih lanjut.
2. Buat Program Intervensi Dini
   - Buat program remedial, bimbingan belajar, atau mentoring untuk mahasiswa dengan capaian akademik rendah di awal studi.
   - Integrasi prediksi dropout dalam proses akademik agar dosen wali dapat melihat potensi risiko lebih awal.
3. Dashboard Risiko Dropout Berkala
   Gunakan dashboard di Looker Studio untuk:
     - Menampilkan mahasiswa berdasarkan skor risiko tertinggi.
     - Memfilter data berdasarkan gender, program studi, atau semester.
     - Menilai efektivitas intervensi dari waktu ke waktu.

Kesimpulan keseluruhan Sistem prediksi dropout berbasis Random Forest yang dibangun dalam proyek ini menjadi alat bantu strategis untuk:
1. Deteksi dini mahasiswa dengan risiko dropout.
2. Pengambilan keputusan akademik dan intervensi yang lebih terarah.
3. Peningkatan kualitas pendidikan melalui monitoring berbasis data.

Dengan alat ini, Jaya Jaya Institut memiliki kesempatan untuk menciptakan lingkungan belajar yang lebih inklusif, adaptif, dan sukses.
