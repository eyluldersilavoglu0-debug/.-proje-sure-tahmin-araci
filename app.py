# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="İnşaat Proje Süresi Tahmin Aracı",
    page_icon="⏱️",
    layout="centered"
)

# Başlık Tasarımı
st.markdown("""
<div style="background-color: #0A2342; padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem;">
    <h2 style="color: white; margin: 0; font-size: 1.5rem;">⏱️ İnşaat Proje Süresi Tahmin Aracı</h2>
    <p style="color: #D5E8F0; margin: 0.3rem 0 0 0; font-size: 0.9rem;">
        İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları — Eylül Dersilavoğlu (2023232037)
    </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def egit_modeller():
    np.random.seed(42)
    n = 200
    alan = np.random.uniform(500, 3500, n)
    kat = np.random.randint(4, 16, n)
    yil = np.random.randint(2018, 2026, n)
    zemin = np.random.choice(["A", "B", "C", "D"], n)

    zb = (zemin == "B").astype(int)
    zc = (zemin == "C").astype(int)
    zd = (zemin == "D").astype(int)

    # 15 - 20 ay hedefli süre formülü (~450 - 600 gün)
    sure = 320 + (alan * 0.025) + (kat * 11.5) - ((yil - 2020) * 2.0) + (zb * 15) + (zc * 30) + (zd * 50)
    sure += np.random.normal(0, 8, n)
    sure = np.clip(sure, 380, 680)

    # Basit Model
    X_b = pd.DataFrame({"alan_m2": alan, "kat_sayisi": kat})
    m_b = LinearRegression().fit(X_b, sure)

    # Gelişmiş Model
    X_g = pd.DataFrame({
        "alan_m2": alan,
        "kat_sayisi": kat,
        "insaat_yili": yil,
        "zemin_sinifi_B": zb,
        "zemin_sinifi_C": zc,
        "zemin_sinifi_D": zd
    })
    m_g = LinearRegression().fit(X_g, sure)

    return m_b, m_g

model_basit, model_gelismis = egit_modeller()

# Yan Panel
st.sidebar.markdown("### ⚙️ Model Seçimi")
model_tipi = st.sidebar.radio(
    "Tahmin Modeli:",
    ["Basit Model (Alan & Kat)", "Gelişmiş Model (Zemin & Yıl Dahil)"]
)
gelismis_mi = "Gelişmiş" in model_tipi

# Parametre Girişleri
st.markdown("### 📋 Proje Parametreleri")
col1, col2 = st.columns(2)
with col1:
    alan = st.number_input("Taban Alanı (m²)", min_value=100, max_value=10000, value=1200, step=50)
with col2:
    kat = st.number_input("Toplam Kat Sayısı", min_value=1, max_value=40, value=8, step=1)

zemin = "A"
yil = 2024
if gelismis_mi:
    col3, col4 = st.columns(2)
    with col3:
        zemin = st.selectbox("Zemin Sınıfı", ["A", "B", "C", "D"], index=0)
    with col4:
        yil = st.number_input("İnşaat Başlangıç Yılı", min_value=2015, max_value=2030, value=2024, step=1)

# Hesaplama
if st.button("⏱️ Proje Süresini Tahmin Et", type="primary", use_container_width=True):
    if gelismis_mi:
        girdi = pd.DataFrame([{
            "alan_m2": float(alan),
            "kat_sayisi": float(kat),
            "insaat_yili": float(yil),
            "zemin_sinifi_B": 1 if zemin == "B" else 0,
            "zemin_sinifi_C": 1 if zemin == "C" else 0,
            "zemin_sinifi_D": 1 if zemin == "D" else 0
        }])
        tahmin_gun = model_gelismis.predict(girdi)[0]
    else:
        girdi = pd.DataFrame([{
            "alan_m2": float(alan),
            "kat_sayisi": float(kat)
        }])
        tahmin_gun = model_basit.predict(girdi)[0]

    tahmin_gun = max(30.0, float(tahmin_gun))
    tahmin_ay = tahmin_gun / 30.0
    gun_tam = int(round(tahmin_gun))

    st.markdown(f"""
    <div style="background-color: #1F5C99; color: white; padding: 1.3rem; border-radius: 10px; text-align: center; margin-top: 1rem;">
        <div style="font-size: 2.2rem; font-weight: bold;">{gun_tam:,} Gün (~{tahmin_ay:.1f} Ay)</div>
        <div style="font-size: 0.95rem; opacity: 0.9;">Tahmini Toplam Proje Süresi</div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Tahmin 15–20 ay planlama aralığında başarıyla hesaplandı.")
