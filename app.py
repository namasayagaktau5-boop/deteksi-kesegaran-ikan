import os
import streamlit as st
from cnn_model import build_cnn_model, prediksi_citra_asli
from fuzzy_mamdani import hitung_kesegaran_fuzzy, inisialisasi_fuzzy
from knn_model import inisialisasi_dan_latih_knn, prediksi_kesegaran_knn

# Load model CNN
model_mata = build_cnn_model('models/eye_model.h5')
model_insang = build_cnn_model('models/gill_model.h5')

# Inisialisasi model Fuzzy & KNN
simulasi_fuzzy = inisialisasi_fuzzy()
model_knn = inisialisasi_dan_latih_knn()

st.title('🐟 Sistem Deteksi Kesegaran Ikan')
st.write('Analisis kesegaran ikan menggunakan CNN dengan pilihan metode klasifikasi **Fuzzy Mamdani** atau **KNN**.')

# Pilihan metode di Sidebar
st.sidebar.header('Pengaturan Metode')
metode_pilihan = st.sidebar.selectbox(
    'Pilih Algoritma Klasifikasi:',
    ['Fuzzy Mamdani', 'K-Nearest Neighbors (KNN)']
)

# Upload gambar mata dan insang
uploaded_mata = st.file_uploader('Pilih Foto Mata Ikan', type=['jpg', 'jpeg', 'png'])
uploaded_insang = st.file_uploader('Pilih Foto Insang Ikan', type=['jpg', 'jpeg', 'png'])

if uploaded_mata is not None and uploaded_insang is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_mata, caption='Foto Mata Ikan', use_column_width=True)
    with col2:
        st.image(uploaded_insang, caption='Foto Insang Ikan', use_column_width=True)

    if st.button('Proses Analisis Kesegaran'):
        path_mata = 'temp_mata.png'
        with open(path_mata, 'wb') as f:
            f.write(uploaded_mata.getbuffer())

        path_insang = 'temp_insang.png'
        with open(path_insang, 'wb') as f:
            f.write(uploaded_insang.getbuffer())

        # Ekstraksi fitur CNN (skala 0 - 100)
        skor_mata = prediksi_citra_asli(model_mata, path_mata) * 100
        skor_insang = prediksi_citra_asli(model_insang, path_insang) * 100

        st.subheader('Hasil Analisis:')
        st.write(f'- Skor Kejernihan Mata: {skor_mata:.2f} / 100')
        st.write(f'- Skor Kemerahan Insang: {skor_insang:.2f} / 100')

        # Logika Berdasarkan Metode yang Dipilih
        if metode_pilihan == 'Fuzzy Mamdani':
            skor_akhir, kategori = hitung_kesegaran_fuzzy(simulasi_fuzzy, skor_mata, skor_insang)
            st.markdown(f'### Metode: Fuzzy Mamdani')
            st.markdown(f'**Skor Kesegaran Akhir:** {skor_akhir:.2f}%')
        else:
            kategori, confidence = prediksi_kesegaran_knn(model_knn, skor_mata, skor_insang)
            st.markdown(f'### Metode: K-Nearest Neighbors (KNN)')
            st.markdown(f'**Tingkat Keyakinan Model:** {confidence:.2f}%')

        # Status Kategori
        if kategori.upper() == 'SEGAR':
            st.success(f'Status Kategori: {kategori}')
        else:
            st.error(f'Status Kategori: {kategori}')