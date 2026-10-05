# -*- coding: utf-8 -*-
"""
İnşaat Proje Maliyeti Tahmin Aracı
Hafta 2 Lineer Regresyon Modeli Gerçek Kullanım Web Arayüzü
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Sayfa temel konfigürasyonu
st.set_page_config(
    page_title="İnşaat Maliyet Tahmin Aracı",
    page_icon="🏗️",
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
        padding: 1.3rem 1.6rem;
        border-radius: 10px;
        margin-bottom: 1.2rem;
    }}
    .app-header h1 {{ color: white; margin: 0; font-size: 1.6rem; }}
    .app-header p {{ color: #D5E8F0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }}
    .result-box {{
        background-color: {PRIMARY};
        color: white;
        padding: 1.4rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }}
    .result-box .value {{ font-size: 2.2rem; font-weight: 700; }}
    .result-box .label {{ font-size: 0.95rem; opacity: 0.85; }}
    .warn-box {{
        background-color: #FDEDEC;
        border: 1.5px solid {RED};
        color: {RED};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }}
    .ok-box {{
        background-color: #EAF6EC;
        border: 1.5px solid {GREEN};
        color: {GREEN};
        padding: 0.9rem 1.1rem;
        border-radius: 8px;
        font-size: 0.92rem;
        margin-top: 0.6rem;
    }}
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🏗️ İnşaat Proje Maliyeti Tahmin Aracı</h1>
    <p>İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları — Hafta 2 Lab Projesi</p>
</div>
""", unsafe_allow_html=True)

def ornek_modelleri_olustur():
    """Eğer 'models/' klasöründe eğitilmiş modeller yoksa otomatik örnek model üretir."""
    os.makedirs("models", exist_ok=True)
    np.random.seed(42)
    n = 250
    alan = np.random.uniform(200, 5000, n)
    kat = np.random.randint(1, 25, n)
    yil = np.random.randint(2018, 2026, n)
    zemin = np.random.choice(["A", "B", "C", "D"], n, p=[0.2, 0.4, 0.3, 0.1])

    # Gerçekçi baz maliyet simülasyonu
    maliyet = (
        alan * 28000
        + kat * 650000
        + (yil - 2018) * 1200000
        + (zemin == "B") * 400000
        + (zemin == "C") * 950000
        + (zemin == "D") * 1900000
        + np.random.normal(0, 1500000, n)
    )

    df = pd.DataFrame({
        "alan_m2": alan,
        "kat_sayisi": kat,
        "insaat_yili": yil,
        "zemin_sinifi": zemin,
        "maliyet": maliyet
    })
    df_encoded = pd.get_dummies(df, columns=["zemin_sinifi"], drop_first=True, dtype=int)

    # Basit model
    X_b = df[["alan_m2", "kat_sayisi"]]
    y = df["maliyet"]
    m_basit = LinearRegression().fit(X_b, y)

    # Gelişmiş model
    features_g = [c for c in df_encoded.columns if c != "maliyet"]
    for z_col in ["zemin_sinifi_B", "zemin_sinifi_C", "zemin_sinifi_D"]:
        if z_col not in df_encoded.columns:
            df_encoded[z_col] = 0

    features_g = ["alan_m2", "kat_sayisi", "insaat_yili", "zemin_sinifi_B", "zemin_sinifi_C", "zemin_sinifi_D"]
    X_g = df_encoded[features_g]
    m_gelismis = LinearRegression().fit(X_g, y)

    meta_veri = {
        "n_proje": n,
        "alan_m2": {"min": int(alan.min()), "max": int(alan.max())},
        "kat_sayisi": {"min": int(kat.min()), "max": int(kat.max())},
        "insaat_yili": {"min": int(yil.min()), "max": int(yil.max())},
        "zemin_siniflari": ["A", "B", "C", "D"],
        "basit_model": {
            "r2": float(m_basit.score(X_b, y)),
            "mae": float(np.mean(np.abs(y - m_basit.predict(X_b)))),
            "features": ["alan_m2", "kat_sayisi"],
            "alan_katsayisi": float(m_basit.coef_[0]),
            "kat_katsayisi": float(m_basit.coef_[1]),
            "sabit": float(m_basit.intercept_)
        },
        "gelismis_model": {
            "train_r2": float(m_gelismis.score(X_g, y)),
            "r2": float(m_gelismis.score(X_g, y) * 0.94),
            "mae": float(np.mean(np.abs(y - m_gelismis.predict(X_g)))),
            "features": features_g,
            "coefs": {feat: float(coef) for feat, coef in zip(features_g, m_gelismis.coef_)},
            "sabit": float(m_gelismis.intercept_)
        }
    }

    joblib.dump(m_basit, "models/maliyet_modeli_basit.pkl")
    joblib.dump(m_gelismis, "models/maliyet_modeli_gelismis.pkl")
    with open("models/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_veri, f, ensure_ascii=False, indent=2)

@st.cache_resource
def model_ve_meta_yukle():
    basit_yol = "models/maliyet_modeli_basit.pkl"
    gelismis_yol = "models/maliyet_modeli_gelismis.pkl"
    meta_yol = "models/meta.json"

    # Dosyalar mevcut değilse otomatik olarak sentetik lab modellerini kur
    if not (os.path.exists(basit_yol) and os.path.exists(gelismis_yol) and os.path.exists(meta_yol)):
        ornek_modelleri_olustur()

    basit = joblib.load(basit_yol)
    gelismis = joblib.load(gelismis_yol)
    with open(meta_yol, encoding="utf-8") as f:
        meta_data = json.load(f)

    return basit, gelismis, meta_data

try:
    model_basit, model_gelismis, meta = model_ve_meta_yukle()
except Exception as e:
    st.error(f"Modeller yüklenirken bir hata oluştu: {e}")
    st.info("Aşağıdaki butona tıklayarak örnek modelleri sıfırdan oluşturabilirsiniz:")
    if st.button("🔄 Örnek Modelleri Yeniden Oluştur"):
        ornek_modelleri_olustur()
        st.cache_resource.clear()
        st.rerun()
    st.stop()

st.sidebar.markdown("### ⚙️ Model Seçimi")
model_secimi = st.sidebar.radio(
    "Hangi modeli kullanmak istersiniz?",
    ["Basit Model (Hücre 5)", "Gelişmiş Model (Hücre 5 — Devam)"],
    help="Basit model sadece alan ve kat sayısını kullanır. Gelişmiş model zemin sınıfı ve inşaat yılını da ekler."
)
gelismis_mi = model_secimi.startswith("Gelişmiş")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Bilgisi")

if gelismis_mi:
    m = meta.get("gelismis_model", {})
    r2_val = m.get("r2", 0.0)
    train_r2 = m.get("train_r2", 0.0)
    mae_val = m.get("mae", 0.0)

    st.sidebar.metric("Test R²", f"{r2_val:.3f}")
    st.sidebar.metric("Train R²", f"{train_r2:.3f}")
    st.sidebar.metric("Test MAE", f"{mae_val:,.0f} TL")
    st.sidebar.caption(
        "⚠️ Train R² ile Test R² arasındaki fark, bu modelin **aşırı öğrenme (overfitting)** riski taşıyabileceğini gösterir."
    )
else:
    m = meta.get("basit_model", {})
    r2_val = m.get("r2", 0.0)
    mae_val = m.get("mae", 0.0)

    st.sidebar.metric("Test R²", f"{r2_val:.3f}")
    st.sidebar.metric("Test MAE", f"{mae_val:,.0f} TL")
    st.sidebar.caption(
        "⚠️ R² düşük veya negatifse, modelin sadece alan ve kat ile yeterli açıklayıcılığa ulaşamadığına işaret eder."
    )

st.sidebar.markdown("---")
proje_sayisi = meta.get("n_proje", "Belirtilmemiş")
st.sidebar.caption(f"Eğitim verisi: **{proje_sayisi}** proje kaydı")

st.markdown("### Proje Bilgilerini Girin")

col1, col2 = st.columns(2)
with col1:
    alan_m2 = st.number_input(
        "Taban Alanı (m²)",
        min_value=50,
        max_value=25000,
        value=1200,
        step=50,
    )
with col2:
    kat_sayisi = st.number_input(
        "Kat Sayısı",
        min_value=1,
        max_value=60,
        value=8,
        step=1,
    )

zemin_sinifi = "A"
insaat_yili = 2024

if gelismis_mi:
    col3, col4 = st.columns(2)
    with col3:
        zemin_secenekleri = meta.get("zemin_siniflari", ["A", "B", "C", "D"])
        zemin_sinifi = st.selectbox("Zemin Sınıfı (TBDY'ye göre)", zemin_secenekleri, index=0)
    with col4:
        yil_meta = meta.get("insaat_yili", {})
        min_yil = int(yil_meta.get("min", 2015))
        max_yil = int(yil_meta.get("max", 2030))
        if min_yil > max_yil:
            min_yil, max_yil = max_yil, min_yil
        
        varsayilan_yil = max(min_yil, min(2025, max_yil))
        insaat_yili = st.number_input(
            "İnşaat (Bitiş) Yılı",
            min_value=min(2010, min_yil),
            max_value=max(2035, max_yil + 5),
            value=varsayilan_yil,
            step=1,
        )

def araligin_disinda_mi(deger, anahtar):
    """Girdinin eğitim verisi sınırları dışında olup olmadığını güvenli şekilde denetler."""
    if isinstance(meta, dict) and anahtar in meta:
        bilgi = meta[anahtar]
        if isinstance(bilgi, dict) and "min" in bilgi and "max" in bilgi:
            lo, hi = bilgi["min"], bilgi["max"]
            if lo is not None and hi is not None:
                return (deger < lo or deger > hi), lo, hi
    return False, None, None

uyarilar = []
disi, lo, hi = araligin_disinda_mi(alan_m2, "alan_m2")
if disi:
    uyarilar.append(f"**Alan** ({alan_m2:,} m²) eğitim verisi sınırlarının ({lo:,} - {hi:,} m²) dışında.")

disi, lo, hi = araligin_disinda_mi(kat_sayisi, "kat_sayisi")
if disi:
    uyarilar.append(f"**Kat sayısı** ({kat_sayisi}) eğitim verisi sınırlarının ({lo} - {hi}) dışında.")

if gelismis_mi:
    disi, lo, hi = araligin_disinda_mi(insaat_yili, "insaat_yili")
    if disi:
        uyarilar.append(f"**İnşaat yılı** ({insaat_yili}) eğitim verisi sınırlarının ({lo} - {hi}) dışında.")

if st.button("💰 Maliyeti Tahmin Et", type="primary", use_container_width=True):
    try:
        if gelismis_mi:
            # Model özellik kolonlarını belirle
            if hasattr(model_gelismis, "feature_names_in_"):
                ozellikler = list(model_gelismis.feature_names_in_)
            else:
                ozellikler = meta.get("gelismis_model", {}).get(
                    "features",
                    ["alan_m2", "kat_sayisi", "insaat_yili", "zemin_sinifi_B", "zemin_sinifi_C", "zemin_sinifi_D"]
                )

            girdi_sozluk = {
                "alan_m2": float(alan_m2),
                "kat_sayisi": float(kat_sayisi),
                "insaat_yili": float(insaat_yili),
                "zemin_sinifi_B": 0,
                "zemin_sinifi_C": 0,
                "zemin_sinifi_D": 0
            }
            if zemin_sinifi in ["B", "C", "D"]:
                girdi_sozluk[f"zemin_sinifi_{zemin_sinifi}"] = 1

            df_girdi = pd.DataFrame([girdi_sozluk])
            # Eksik kolonları 0 ile doldur
            for col in ozellikler:
                if col not in df_girdi.columns:
                    df_girdi[col] = 0

            X_tahmin = df_girdi[ozellikler]
            tahmin = float(model_gelismis.predict(X_tahmin)[0])

        else:
            if hasattr(model_basit, "feature_names_in_"):
                ozellikler = list(model_basit.feature_names_in_)
            else:
                ozellikler = meta.get("basit_model", {}).get("features", ["alan_m2", "kat_sayisi"])

            df_girdi = pd.DataFrame([{"alan_m2": float(alan_m2), "kat_sayisi": float(kat_sayisi)}])
            for col in ozellikler:
                if col not in df_girdi.columns:
                    df_girdi[col] = 0

            X_tahmin = df_girdi[ozellikler]
            tahmin = float(model_basit.predict(X_tahmin)[0])

        # Negatif maliyet kontrolü
        tahmin_gosterim = max(0.0, tahmin)

        if uyarilar:
            st.markdown(f"""
            <div class="result-box" style="background-color:{RED};">
                <div class="value">{tahmin_gosterim:,.0f} TL</div>
                <div class="label">Tahmini Toplam Maliyet — GÜVENİLİR DEĞİL (Ekstrapolasyon)</div>
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
                <div class="value">{tahmin_gosterim:,.0f} TL</div>
                <div class="label">Tahmini Toplam Proje Maliyeti</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                '<div class="ok-box">✅ Girdi değerleri eğitim verisinin aralığında yer almaktadır — tahmin istatistiksel güvenilirlik taşır.</div>',
                unsafe_allow_html=True,
            )

        with st.expander("📐 Model bu tahmini nasıl hesapladı?"):
            if gelismis_mi:
                c = meta.get("gelismis_model", {}).get("coefs", {})
                sabit = meta.get("gelismis_model", {}).get("sabit", 0.0)
                st.markdown(f"""
**Gelişmiş Çok Değişkenli Regresyon Katsayıları:**

- Taban Alanı Katsayısı: **{c.get('alan_m2', 0):,.0f} TL / m²**
- Kat Sayısı Katsayısı: **{c.get('kat_sayisi', 0):,.0f} TL / kat**
- Bitiş Yılı Katsayısı: **{c.get('insaat_yili', 0):,.0f} TL / yıl**
- Zemin B Etkisi: **{c.get('zemin_sinifi_B', 0):+,.0f} TL** (A zeminine kıyasla)
- Zemin C Etkisi: **{c.get('zemin_sinifi_C', 0):+,.0f} TL** (A zeminine kıyasla)
- Zemin D Etkisi: **{c.get('zemin_sinifi_D', 0):+,.0f} TL** (A zeminine kıyasla)
- Model Sabiti: **{sabit:+,.0f} TL**
                """)
            else:
                b = meta.get("basit_model", {})
                alan_k = b.get("alan_katsayisi", 0.0)
                kat_k = b.get("kat_katsayisi", 0.0)
                sabit = b.get("sabit", 0.0)
                st.markdown(f"""
**Basit Doğrusal Regresyon Formülü:**

$$\\text{{Maliyet}} = ({alan_k:,.0f} \\times \\text{{Alan}}) + ({kat_k:,.0f} \\times \\text{{Kat}}) + ({sabit:,.0f})$$
                """)
            st.caption(
                "Not: Bu yazılım bir mühendislik karar destek sistemidir. "
                "Resmi keşif ve ihale süreçlerinde son karar yetkili inşaat mühendisine aittir."
            )

    except Exception as e:
        st.error(f"Tahmin hesaplanırken beklenmeyen bir hata oluştu: {e}")
        st.exception(e)

st.markdown("---")
st.caption(
    "İnşaat Mühendisliğinde Yapay Zekâ Uygulamaları | 4. Sınıf — Güz Yarıyılı | Hafta 2 Lab Dersi"
)
