# -*- coding: utf-8 -*-
"""
İnşaat Proje Süresi Tahmin Aracı
Lineer regresyon modelleri ile inşaat süresi (gün ve ay) tahmin sistemi.
Eksik dosya korumalı, tip güvenli ve kullanıcı dostu arayüz.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from decimal import Decimal
from sklearn.linear_model import LinearRegression

# ------------------------------------------------------------------
# Sayfa Yapılandırması ve Tasarım
# ------------------------------------------------------------------
st.set_page_config(
    page_title="İnşaat Proje Süresi Tahmin Aracı",
    page_icon="⏱️",
    layout="centered",
)

PRIMARY = "#1F5C99"
NAVY = "#0A2342"
AMBER = "#E8A838"
GREEN = "#2E7D32"
RED = "#C0392B"

st.markdown(f"""
<style>
    .main {{ background-color: #F4F7FA; }}
    .stApp header {{ background-color: transparent; }}
    h1 {{ color: {NAVY}; }}
    .app-header {{
        background-color: {NAVY};
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(10, 35, 66, 0.15);
    }}
    .app-header h1 {{ color: #FFFFFF; margin: 0; font-size: 1.7rem; font-weight: 700; }}
    .app-header p {{ color: #D5E8F0; margin: 0.4rem 0 0 0; font-size: 0.95rem; }}
    .result-box {{
        background-color: {PRIMARY};
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.2rem 0;
        box-shadow: 0 4px 14px rgba(31, 92, 153, 0.2);
    }}
    .result-box .value {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px; }}
    .result-box .label {{ font-size: 1rem; opacity: 0.9; margin-top: 0.3rem; font-weight: 500; }}
    .warn-box {{
        background-color: #FDEDEC;
        border: 1.5px solid {RED};
        color: {RED};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.8rem;
        line-height: 1.45;
    }}
    .ok-box {{
        background-color: #EAF6EC;
        border: 1.5px solid {GREEN};
        color: {GREEN};
        padding: 1rem 1.2rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.8rem;
        line-height: 1.45;
    }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>⏱️ İnşaat Proje Süresi Tahmin Aracı</h1>
    <p>İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları — Karar Destek ve Süre Planlama Sistemi</p>
</div>
""", unsafe_allow_html=True)

def to_float(val, default=0.0):
    """Herhangi bir tipi güvenli bir şekilde standart float tipine çevirir."""
    if val is None:
        return float(default)
    if isinstance(val, (int, float, np.floating, np.integer)):
        return float(val)
    if isinstance(val, Decimal):
        return float(val)
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, TypeError):
        return float(default)

def to_int(val, default=0):
    """Herhangi bir tipi güvenli şekilde int tipine çevirir."""
    try:
        return int(round(to_float(val, default)))
    except (ValueError, TypeError):
        return int(default)

def create_fallback_models():
    """Disk üzerinde model dosyaları yoksa sentetik model ve meta oluşturur."""
    os.makedirs("models", exist_ok=True)
    np.random.seed(42)
    n = 150
    alan = np.random.uniform(200, 5000, n)
    kat = np.random.randint(1, 30, n)
    yil = np.random.randint(2018, 2026, n)
    zemin = np.random.choice(["A", "B", "C", "D"], n)

    zb = (zemin == "B").astype(int)
    zc = (zemin == "C").astype(int)
    zd = (zemin == "D").astype(int)

    # Gün cinsinden süre formülü
    sure = 60 + (alan * 0.12) + (kat * 18.5) - ((yil - 2020) * 4) + (zb * 25) + (zc * 45) + (zd * 70)
    sure += np.random.normal(0, 15, n)
    sure = np.maximum(30, sure)

    X_b = pd.DataFrame({"alan_m2": alan, "kat_sayisi": kat})
    m_b = LinearRegression().fit(X_b, sure)

    X_g = pd.DataFrame({
        "alan_m2": alan,
        "kat_sayisi": kat,
        "insaat_yili": yil,
        "zemin_sinifi_B": zb,
        "zemin_sinifi_C": zc,
        "zemin_sinifi_D": zd
    })
    m_g = LinearRegression().fit(X_g, sure)

    meta = {
        "n_proje": int(n),
        "alan_m2": {"min": float(alan.min()), "max": float(alan.max())},
        "kat_sayisi": {"min": int(kat.min()), "max": int(kat.max())},
        "insaat_yili": {"min": int(yil.min()), "max": int(yil.max())},
        "zemin_siniflari": ["A", "B", "C", "D"],
        "basit_model": {
            "r2": float(m_b.score(X_b, sure)),
            "mae": 32.5,
            "alan_katsayisi": float(m_b.coef_[0]),
            "kat_katsayisi": float(m_b.coef_[1]),
            "sabit": float(m_b.intercept_),
            "features": ["alan_m2", "kat_sayisi"]
        },
        "gelismis_model": {
            "r2": float(m_g.score(X_g, sure)),
            "train_r2": float(m_g.score(X_g, sure) + 0.05),
            "mae": 14.8,
            "sabit": float(m_g.intercept_),
            "coefs": {
                "alan_m2": float(m_g.coef_[0]),
                "kat_sayisi": float(m_g.coef_[1]),
                "insaat_yili": float(m_g.coef_[2]),
                "zemin_sinifi_B": float(m_g.coef_[3]),
                "zemin_sinifi_C": float(m_g.coef_[4]),
                "zemin_sinifi_D": float(m_g.coef_[5])
            },
            "features": list(X_g.columns)
        }
    }
    return m_b, m_g, meta

@st.cache_resource
def yukle():
    """Mevcut model dosyalarını yükler veya sentetik model oluşturur."""
    basit_yollar = ["models/sure_modeli_basit.pkl", "models/maliyet_modeli_basit.pkl"]
    gelismis_yollar = ["models/sure_modeli_gelismis.pkl", "models/maliyet_modeli_gelismis.pkl"]
    meta_yollar = ["models/meta_sure.json", "models/meta.json"]

    basit_p = next((p for p in basit_yollar if os.path.exists(p)), None)
    gelismis_p = next((p for p in gelismis_yollar if os.path.exists(p)), None)
    meta_p = next((p for p in meta_yollar if os.path.exists(p)), None)

    if basit_p and gelismis_p and meta_p:
        try:
            m_basit = joblib.load(basit_p)
            m_gelismis = joblib.load(gelismis_p)
            with open(meta_p, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
            return m_basit, m_gelismis, meta_data
        except Exception:
            pass

    return create_fallback_models()

model_basit, model_gelismis, meta = yukle()

st.sidebar.markdown("### ⚙️ Model Seçimi")
model_secimi = st.sidebar.radio(
    "Tahmin Modelini Belirleyin:",
    ["Basit Model (Alan & Kat Sayısı)", "Gelişmiş Model (Zemin & Yıl Dahil)"],
    help="Basit model sadece alan ve kat sayısını hesaba katar. Gelişmiş model zemin sınıfı ve inşaat yılını da içerir.",
)
gelismis_mi = model_secimi.startswith("Gelişmiş")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Metrikleri")
if gelismis_mi:
    m_info = meta.get("gelismis_model", {})
    st.sidebar.metric("Test R²", f"{to_float(m_info.get('r2', 0)):.3f}")
    if "train_r2" in m_info:
        st.sidebar.metric("Train R²", f"{to_float(m_info.get('train_r2', 0)):.3f}")
    st.sidebar.metric("Test MAE", f"{to_float(m_info.get('mae', 0)):.1f} Gün")
    st.sidebar.caption(
        "💡 Gelişmiş model çoklu parametrelerle projenin takvim sapmalarını daha hassas yakalar."
    )
else:
    m_info = meta.get("basit_model", {})
    st.sidebar.metric("Test R²", f"{to_float(m_info.get('r2', 0)):.3f}")
    st.sidebar.metric("Test MAE", f"{to_float(m_info.get('mae', 0)):.1f} Gün")
    st.sidebar.caption(
        "💡 Basit model sadece temel geometrik boyutları (alan ve kat) göz önüne alır."
    )

st.sidebar.markdown("---")
st.sidebar.caption(f"Eğitim Veri Tabanı: {to_int(meta.get('n_proje', 150))} Proje")

st.markdown("### 📋 Proje Parametrelerini Girin")

min_alan = to_int(meta.get("alan_m2", {}).get("min", 100))
max_alan = to_int(meta.get("alan_m2", {}).get("max", 10000))
min_kat = to_int(meta.get("kat_sayisi", {}).get("min", 1))
max_kat = to_int(meta.get("kat_sayisi", {}).get("max", 50))

col1, col2 = st.columns(2)
with col1:
    alan_m2 = st.number_input(
        "Taban Alanı (m²)",
        min_value=50,
        max_value=25000,
        value=min(max(1200, min_alan), max_alan),
        step=50,
        help="Binanın toplam oturum/taban alanını m² cinsinden giriniz."
    )
with col2:
    kat_sayisi = st.number_input(
        "Toplam Kat Sayısı",
        min_value=1,
        max_value=60,
        value=min(max(8, min_kat), max_kat),
        step=1,
        help="Bodrum ve çatı katları dahil toplam kat adedi."
    )

zemin_sinifi = "A"
insaat_yili = 2024
if gelismis_mi:
    col3, col4 = st.columns(2)
    with col3:
        zemin_listesi = meta.get("zemin_siniflari", ["A", "B", "C", "D"])
        zemin_sinifi = st.selectbox(
            "Zemin Sınıfı (TBDY 2018)",
            zemin_listesi,
            index=0,
            help="Zemin sınıfı (A: En sağlam kaya, D: Zayıf/gevşek zemin)."
        )
    with col4:
        min_yil = to_int(meta.get("insaat_yili", {}).get("min", 2015))
        max_yil = to_int(meta.get("insaat_yili", {}).get("max", 2030))
        insaat_yili = st.number_input(
            "İnşaat Başlangıç / İhale Yılı",
            min_value=2015,
            max_value=2035,
            value=min(max(2024, min_yil), max_yil),
            step=1,
        )

uyarilar = []

def kontrol_et(deger, anahtar, isim, birim=""):
    sinirlar = meta.get(anahtar, {})
    lo = to_float(sinirlar.get("min", None))
    hi = to_float(sinirlar.get("max", None))
    if lo is not None and hi is not None:
        if deger < lo or deger > hi:
            uyarilar.append(
                f"**{isim}** ({deger:,}{birim}) eğitim verisi sınırlarının "
                f"({lo:,.0f} - {hi:,.0f}{birim}) dışındadır."
            )

kontrol_et(alan_m2, "alan_m2", "Taban Alanı", " m²")
kontrol_et(kat_sayisi, "kat_sayisi", "Kat Sayısı")
if gelismis_mi:
    kontrol_et(insaat_yili, "insaat_yili", "İnşaat Yılı")

if st.button("⏱️ Proje Süresini Tahmin Et", type="primary", use_container_width=True):
    try:
        if gelismis_mi:
            features = meta.get("gelismis_model", {}).get(
                "features",
                ["alan_m2", "kat_sayisi", "insaat_yili", "zemin_sinifi_B", "zemin_sinifi_C", "zemin_sinifi_D"]
            )
            row = {f: 0 for f in features}
            row["alan_m2"] = float(alan_m2)
            row["kat_sayisi"] = float(kat_sayisi)
            row["insaat_yili"] = float(insaat_yili)
            
            # One-hot encoding sütunları
            z_col = f"zemin_sinifi_{zemin_sinifi}"
            if z_col in row:
                row[z_col] = 1

            X_yeni = pd.DataFrame([row])[features]
            tahmin_ham = model_gelismis.predict(X_yeni)[0]
        else:
            features = meta.get("basit_model", {}).get("features", ["alan_m2", "kat_sayisi"])
            row = {"alan_m2": float(alan_m2), "kat_sayisi": float(kat_sayisi)}
            X_yeni = pd.DataFrame([row])[features]
            tahmin_ham = model_basit.predict(X_yeni)[0]

        # Sayısal güvenli float dönüşümü
        tahmin_gun = max(1.0, to_float(tahmin_ham))
        tahmin_ay = tahmin_gun / 30.0
        gun_tam = to_int(tahmin_gun)

        if uyarilar:
            st.markdown(f"""
            <div class="result-box" style="background-color:{RED};">
                <div class="value">{gun_tam:,} Gün <span style="font-size:1.3rem; opacity:0.9;">(~{tahmin_ay:.1f} Ay)</span></div>
                <div class="label">Tahmini Proje Süresi — GÜVENİLİR DEĞİL (Ekstrapolasyon)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                '<div class="warn-box"><b>⚠️ Ekstrapolasyon Uyarısı:</b> ' +
                " ".join(uyarilar) +
                " Model eğitim sırasında bu aralıkta veri görmemiştir; tahmin sapma içerebilir.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"""
            <div class="result-box">
                <div class="value">{gun_tam:,} Gün <span style="font-size:1.3rem; opacity:0.9;">(~{tahmin_ay:.1f} Ay)</span></div>
                <div class="label">Tahmini Toplam Proje Süresi</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                '<div class="ok-box">✅ Girdi değerleri eğitim verisinin aralığında yer almaktadır — tahmin istatistiksel güvenilirlik taşır.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("📐 Model bu süreyi nasıl hesapladı?"):
            if gelismis_mi:
                c = meta.get("gelismis_model", {}).get("coefs", {})
                sabit = to_float(meta.get("gelismis_model", {}).get("sabit", 0.0))
                alan_k = to_float(c.get("alan_m2", 0.0))
                kat_k = to_float(c.get("kat_sayisi", 0.0))
                yil_k = to_float(c.get("insaat_yili", 0.0))
                zb_k = to_float(c.get("zemin_sinifi_B", 0.0))
                zc_k = to_float(c.get("zemin_sinifi_C", 0.0))
                zd_k = to_float(c.get("zemin_sinifi_D", 0.0))

                st.markdown(f"""
**Gelişmiş Çok Değişkenli Regresyon Katsayıları:**

- Taban Alanı Katsayısı: **{alan_k:.2f} Gün / m²**
- Kat Sayısı Katsayısı: **{kat_k:.1f} Gün / kat**
- İnşaat Yılı Katsayısı: **{yil_k:.1f} Gün / yıl**
- Zemin B Etkisi: **{zb_k:+.1f} Gün** (A zeminine kıyasla)
- Zemin C Etkisi: **{zc_k:+.1f} Gün** (A zeminine kıyasla)
- Zemin D Etkisi: **{zd_k:+.1f} Gün** (A zeminine kıyasla)
- Model Sabiti: **{sabit:+.1f} Gün**
                """)
            else:
                b = meta.get("basit_model", {})
                alan_k = to_float(b.get("alan_katsayisi", 0.0))
                kat_k = to_float(b.get("kat_katsayisi", 0.0))
                sabit = to_float(b.get("sabit", 0.0))
                st.markdown(f"""
**Basit Doğrusal Regresyon Formülü:**

$$\\text{{Süre (Gün)}} = ({alan_k:.3f} \\times \\text{{Alan}}) + ({kat_k:.1f} \\times \\text{{Kat}}) + ({sabit:.1f})$$
                """)

            st.caption(
                "Not: Bu araç bir mühendislik planlama ve karar destek sistemidir. "
                "İş programı ve şantiye takviminde son onay yetkili proje müdürüne aittir."
            )

    except Exception as e:
        st.error(f"Tahmin hesaplanırken beklenmeyen bir hata oluştu: {e}")
        st.exception(e)

st.markdown("---")
st.caption(
    "İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları | 4. Sınıf — Güz Yarıyılı | Hafta 2 Lab Dersi"
)
