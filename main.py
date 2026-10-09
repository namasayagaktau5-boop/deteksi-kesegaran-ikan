import os
from cnn_model import build_cnn_model, prediksi_citra_asli
from fuzzy_mamdani import inisialisasi_fuzzy, hitung_kesegaran_fuzzy

def main():
    print("=" * 60)
    print(" SISTEM PENDETEKSI KESEGARAN IKAN (CNN + FUZZY MAMDANI)")
    print("=" * 60)

    print("[1] Memuat Model CNN...")
    model_mata = build_cnn_model()
    model_insang = build_cnn_model()

    print("[2] Memuat Sistem Fuzzy Mamdani...")
    simulasi_fuzzy = inisialisasi_fuzzy()

    # Tentukan path file foto dari folder dataset Anda
    path_mata = os.path.join('mata', 'tidak_segar', 'image.png')
    path_insang = os.path.join('insang', 'segar', 'image.png')

    print(f"\n[3] Memproses Foto Asli:")
    print(f"    - Foto Mata   : {path_mata}")
    print(f"    - Foto Insang : {path_insang}")

    # Pengolahan citra nyata oleh CNN
    skor_mata = prediksi_citra_asli(model_mata, path_mata)
    skor_insang = prediksi_citra_asli(model_insang, path_insang)

    print(f"\n    -> Hasil Skor Kejernihan Mata   : {skor_mata:.2f} / 100")
    print(f"    -> Hasil Skor Kemerahan Insang  : {skor_insang:.2f} / 100")

    print("\n[4] Pengambilan Keputusan Akhir dengan Fuzzy Logic Mamdani...")
    skor_akhir, kategori = hitung_kesegaran_fuzzy(simulasi_fuzzy, skor_mata, skor_insang)

    print("\n" + "=" * 60)
    print(" HASIL ANALISIS KESEGARAN IKAN")
    print("=" * 60)
    print(f" Skor Kesegaran Akhir : {skor_akhir:.2f}%")
    print(f" Status Kategori      : {kategori}")
    print("=" * 60)

if __name__ == "__main__":
    main()