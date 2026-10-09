import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def inisialisasi_fuzzy():
    # Variabel Input & Output (Skala 0 - 100)
    skor_mata = ctrl.Antecedent(np.arange(0, 101, 1), 'skor_mata')
    skor_insang = ctrl.Antecedent(np.arange(0, 101, 1), 'skor_insang')
    kesegaran = ctrl.Consequent(np.arange(0, 101, 1), 'kesegaran')

    # Fungsi Keanggotaan (Membership Function)
    skor_mata['buram'] = fuzz.trimf(skor_mata.universe, [0, 0, 50])
    skor_mata['sedang'] = fuzz.trimf(skor_mata.universe, [30, 50, 70])
    skor_mata['jernih'] = fuzz.trimf(skor_mata.universe, [50, 100, 100])

    skor_insang['pucat'] = fuzz.trimf(skor_insang.universe, [0, 0, 50])
    skor_insang['sedang'] = fuzz.trimf(skor_insang.universe, [30, 50, 70])
    skor_insang['merah'] = fuzz.trimf(skor_insang.universe, [50, 100, 100])

    kesegaran['tidak_segar'] = fuzz.trimf(kesegaran.universe, [0, 0, 45])
    kesegaran['kurang_segar'] = fuzz.trimf(kesegaran.universe, [35, 50, 65])
    kesegaran['segar'] = fuzz.trimf(kesegaran.universe, [55, 100, 100])

    # Rule Base Lengkap (9 Kombinasi)
    rule1 = ctrl.Rule(skor_mata['buram'] & skor_insang['pucat'], kesegaran['tidak_segar'])
    rule2 = ctrl.Rule(skor_mata['buram'] & skor_insang['sedang'], kesegaran['tidak_segar'])
    rule3 = ctrl.Rule(skor_mata['buram'] & skor_insang['merah'], kesegaran['kurang_segar'])
    rule4 = ctrl.Rule(skor_mata['sedang'] & skor_insang['pucat'], kesegaran['tidak_segar'])
    rule5 = ctrl.Rule(skor_mata['sedang'] & skor_insang['sedang'], kesegaran['kurang_segar'])
    rule6 = ctrl.Rule(skor_mata['sedang'] & skor_insang['merah'], kesegaran['kurang_segar'])
    rule7 = ctrl.Rule(skor_mata['jernih'] & skor_insang['pucat'], kesegaran['kurang_segar'])
    rule8 = ctrl.Rule(skor_mata['jernih'] & skor_insang['sedang'], kesegaran['kurang_segar'])
    rule9 = ctrl.Rule(skor_mata['jernih'] & skor_insang['merah'], kesegaran['segar'])

    sistem_kontrol = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    return ctrl.ControlSystemSimulation(sistem_kontrol)

def hitung_kesegaran_fuzzy(simulasi, mata_score, insang_score):
    simulasi.input['skor_mata'] = mata_score
    simulasi.input['skor_insang'] = insang_score
    simulasi.compute()
    
    skor_akhir = simulasi.output['kesegaran']
    if skor_akhir >= 60:
        kategori = "SEGAR"
    elif skor_akhir >= 40:
        kategori = "KURANG SEGAR"
    else:
        kategori = "TIDAK SEGAR"
        
    return skor_akhir, kategori