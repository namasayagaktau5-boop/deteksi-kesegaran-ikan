import os
import streamlit as st
from cnn_model import build_cnn_model, prediksi_citra_asli
from fuzzy_mamdani import hitung_kesegaran_fuzzy, inisialisasi_fuzzy

# Load model CNN yang sudah dilatih
model_mata = build_cnn_model('../models/eye_model.h5')
model_insang = build_cnn_model('../models/gill_model.h5')

# Inisialisasi objek simulasi fuzzy
simulasi_fuzzy = inisialisasi_fuzzy()

st.title('🐟 Sistem Deteksi Kesegaran Ikan')
st.write(
    'Upload foto mata dan insang ikan untuk mendeteksi tingkat'
    ' kesegarannya secara otomatis menggunakan CNN & Fuzzy Mamdani.'
)

# Upload gambar mata
uploaded_mata = st.file_uploader(
    'Pilih Foto Mata Ikan', type=['jpg', 'jpeg', 'png']
)
# Upload gambar insang
uploaded_insang = st.file_uploader(
    'Pilih Foto Insang Ikan', type=['jpg', 'jpeg', 'png']
)

if uploaded_mata is not None and uploaded_insang is not None:
  # Tampilkan gambar yang di-upload
  col1, col2 = st.columns(2)
  with col1:
    st.image(uploaded_mata, caption='Foto Mata Ikan', width='stretch')
  with col2:
    st.image(uploaded_insang, caption='Foto Insang Ikan', width='stretch')

  if st.button('Proses Analisis Kesegaran'):
    # Simpan sementara file yang di-upload
    path_mata = 'temp_mata.png'
    with open(path_mata, 'wb') as f:
      f.write(uploaded_mata.getbuffer())

    path_insang = 'temp_insang.png'
    with open(path_insang, 'wb') as f:
      f.write(uploaded_insang.getbuffer())

    # Proses prediksi dengan CNN (dikalikan 100 agar dalam rentang skala 0-100)
    skor_mata = prediksi_citra_asli(model_mata, path_mata) * 100
    skor_insang = prediksi_citra_asli(model_insang, path_insang) * 100

    # Proses dengan Fuzzy Mamdani menggunakan 3 argumen yang diperlukan
    skor_akhir, kategori = hitung_kesegaran_fuzzy(
        simulasi_fuzzy, skor_mata, skor_insang
    )

    # Tampilkan Hasil Analisis
    st.subheader('Hasil Analisis:')
    st.write(f'- Skor Kejernihan Mata: {skor_mata:.2f} / 100')
    st.write(f'- Skor Kemerahan Insang: {skor_insang:.2f} / 100')
    st.markdown(f'### Skor Kesegaran Akhir: {skor_akhir:.2f}%')

    if kategori.upper() == 'SEGAR':
      st.success(f'Status Kategori: {kategori}')
    else:
      st.error(f'Status Kategori: {kategori}')