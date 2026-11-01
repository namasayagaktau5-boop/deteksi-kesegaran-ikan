import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def inisialisasi_dan_latih_knn():
    # Data latih berdasarkan [skor_mata, skor_insang] (skala 0-100)
    # Label: 1 = Segar, 0 = Tidak Segar
    X_train = np.array([
        [90, 85], [95, 92], [85, 88], [88, 90],  # Segar
        [20, 15], [30, 25], [15, 20], [25, 30]   # Tidak Segar
    ])
    y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    # Inisialisasi model KNN dengan K=3
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

def prediksi_kesegaran_knn(knn_model, skor_mata, skor_insang):
    X_test = np.array([[skor_mata, skor_insang]])
    prediksi = knn_model.predict(X_test)[0]
    probabilitas = knn_model.predict_proba(X_test)[0]

    kategori = "SEGAR" if prediksi == 1 else "TIDAK SEGAR"
    skor_keyakinan = max(probabilitas) * 100
    return kategori, skor_keyakinan