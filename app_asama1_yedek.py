# -*- coding: utf-8 -*-
"""
Elektrikli Araç Şarj Talebi ve Şebeke Kapasitesi Karar Destek Sistemi (KDS)
Aşama 1: B0 Noktasal Yük Tahmini ve S1 Belirsizlik Üst Sınırı Motoru

Origin: 31.08.2022 23:00
Hedef: 01.09.2022 00:00 - 23:00 (24 saatlik ufuk)
Metodoloji: 168 hücrelik (gün x saat) B0 profil ölçekleme, 2 günlük BOSLUK tamponu,
            ve son 8 haftalık kalibrasyon artıklarıyla hesaplanan q=0.90 S1 karekök üst sınırı.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="EV Şarj KDS - Aşama 1: B0 & S1 Tahmin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Dizin ve Sabitler
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FE_DIR = BASE_DIR / "outputs_feature_engineering"
LP_PATH = BASE_DIR / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"

TEST_BASLANGIC = pd.Timestamp("2022-09-01 00:00:00")
PROFIL_BASLANGIC = pd.Timestamp("2021-10-07 18:00:00")
ORIGIN_DATETIME = pd.Timestamp("2022-08-31 23:00:00")
BOSLUK = pd.Timedelta(days=2)
KAL_HAFTA = 8
Q_LEVEL = 0.90
SEVIYE_COL = "started_session_energy_kwh_rolling_mean_168"

SCENARIOS = {
    "U100 (Yönetimsiz / Temel)": "yuk_U100_kwh",
    "F50 (50 kWh Esneklik)": "yuk_F50_kwh",
    "F25 (25 kWh Esneklik)": "yuk_F25_kwh"
}

# -----------------------------------------------------------------------------
# Veri Yükleme ve Önbellekleme
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Veri setleri ve geçmiş seriler yükleniyor...")
def verileri_yukle():
    """
    Eğitim, doğrulama ve yaklaşık yük CSV'lerini güvenli şekilde yükler.
    Eylül 2022 test verisi sızıntısını kesinlikle engeller.
    """
    train_cols = ["origin_datetime", "target_datetime", "forecast_horizon",
                  "target_weekday", "target_hour", SEVIYE_COL, "started_session_energy_kwh"]
    
    train = pd.read_csv(FE_DIR / "train_verisi.csv", usecols=train_cols,
                        parse_dates=["origin_datetime", "target_datetime"])
    val = pd.read_csv(FE_DIR / "validation_verisi.csv", usecols=train_cols,
                      parse_dates=["origin_datetime", "target_datetime"])
    
    tum = pd.concat([train, val], ignore_index=True)
    assert tum.target_datetime.max() < TEST_BASLANGIC, "Kritik Hata: Test verisi modele karışmış olamaz!"
    
    yuk = pd.read_csv(LP_PATH, encoding="utf-8-sig", parse_dates=["datetime"], index_col="datetime")
    yuk = yuk[yuk.index < TEST_BASLANGIC]
    
    # Origin seviyesi için başlangıç serisi (son 168 saat ortalaması)
    bas_seri = (tum.drop_duplicates("target_datetime")
                .set_index("target_datetime")["started_session_energy_kwh"]
                .sort_index())
    
    return tum, yuk, bas_seri

# -----------------------------------------------------------------------------
# B0 ve S1 Hesaplama Fonksiyonları
# -----------------------------------------------------------------------------
def profil_hesapla(seri: pd.Series, bitis_tarihi: pd.Timestamp):
    """
    Haftanın günü x saat bazında 168 hücrelik ortalama yük profilini ve genel ortalamayı üretir.
    Veri aralığı: [PROFIL_BASLANGIC, bitis_tarihi)
    """
    s = seri[(seri.index < bitis_tarihi) & (seri.index >= PROFIL_BASLANGIC)]
    d = pd.DataFrame({"v": s.to_numpy(), "wd": s.index.dayofweek, "hr": s.index.hour})
    prof = d.groupby(["wd", "hr"])["v"].mean().rename("profil").reset_index()
    genel_ortalama = float(s.mean())
    return prof, genel_ortalama


def b0_tahmini(ev_df: pd.DataFrame, seri: pd.Series, bitis_tarihi: pd.Timestamp):
    """
    ev_df içindeki her satır için 168 hücrelik profil ve seviye çarpımı ile B0 tahmini üretir.
    b0 = (profil[wd, hr] / genel_ortalama) * seviye
    """
    prof, genel = profil_hesapla(seri, bitis_tarihi)
    pv = ev_df[["target_weekday", "target_hour"]].merge(
        prof, left_on=["target_weekday", "target_hour"], right_on=["wd", "hr"], how="left"
    )["profil"].to_numpy()
    
    tahmin = (pv / genel) * ev_df[SEVIYE_COL].to_numpy()
    return np.clip(tahmin, 0.0, None)


@st.cache_data(show_spinner="S1 kalibrasyon katsayıları hesaplanıyor...")
def kalibrasyon_hesapla(sc_key: str):
    """
    Hedef dönemin 2 gün öncesi (kesme) ve geriye doğru 8 haftalık kalibrasyon penceresi
    üzerinde artık standartlaştırması yaparak q=0.90 katsayısını hesaplar.
    Veri sızıntısını önlemek için kalibrasyon profili cal_bas öncesi verilerle oluşturulur.
    """
    tum, yuk, _ = verileri_yukle()
    kolon = SCENARIOS[sc_key]
    seri = yuk[kolon]
    
    kesme = TEST_BASLANGIC - BOSLUK  # 2022-08-30 00:00:00
    cal_bas = kesme - pd.Timedelta(weeks=KAL_HAFTA)  # 2022-07-05 00:00:00
    
    cal = tum[(tum.target_datetime >= cal_bas) & (tum.target_datetime < kesme)]
    
    # Kalibrasyon profili cal_bas öncesinden türetilir (kesinlikle kalibrasyon verisi sızmaz)
    b_cal = b0_tahmini(cal, seri, cal_bas)
    
    a_cal = seri.reindex(pd.DatetimeIndex(cal.target_datetime)).to_numpy()
    m = float(a_cal.mean())
    
    ol_cal = np.sqrt(b_cal + 0.05 * m)
    z = (a_cal - b_cal) / ol_cal
    z_q = float(np.quantile(z, Q_LEVEL))
    
    return {
        "z_q": z_q,
        "m": m,
        "cal_bas": cal_bas,
        "kesme": kesme,
        "n_cal": len(cal),
        "z_min": float(z.min()),
        "z_max": float(z.max())
    }


def tahmin_24h_uret(sc_key: str):
    """
    31.08.2022 23:00 origin'i için 01.09.2022 tarihindeki 24 saatlik B0 ve S1 değerlerini hesaplar.
    """
    tum, yuk, bas_seri = verileri_yukle()
    kolon = SCENARIOS[sc_key]
    seri = yuk[kolon]
    
    # Origin anında bilinen son 168 saatlik başlangıç enerjisi ortalaması
    origin_seviye = float(bas_seri.loc[:ORIGIN_DATETIME].tail(168).mean())
    
    target_dt = pd.date_range("2022-09-01 00:00:00", periods=24, freq="h")
    ev_test = pd.DataFrame({
        "target_datetime": target_dt,
        "forecast_horizon": range(1, 25),
        "target_weekday": target_dt.dayofweek,
        "target_hour": target_dt.hour,
        SEVIYE_COL: origin_seviye
    })
    
    kesme = TEST_BASLANGIC - BOSLUK  # 2022-08-30 00:00:00
    b_test = b0_tahmini(ev_test, seri, kesme)
    
    cal_info = kalibrasyon_hesapla(sc_key)
    m = cal_info["m"]
    z_q = cal_info["z_q"]
    
    ol_test = np.sqrt(b_test + 0.05 * m)
    s1_test = np.clip(b_test + z_q * ol_test, 0.0, None)
    
    df_sonuc = pd.DataFrame({
        "Zaman": target_dt,
        "Saat": [f"{h:02d}:00" for h in target_dt.hour],
        "B0_Tahmin_kW": np.round(b_test, 2),
        "S1_Ust_Sinir_kW": np.round(s1_test, 2),
        "Guvenlik_Payi_kW": np.round(s1_test - b_test, 2),
        "Olcek_ol": np.round(ol_test, 2)
    })
    
    return df_sonuc, origin_seviye, cal_info

# -----------------------------------------------------------------------------
# Arayüz Bileşenleri
# -----------------------------------------------------------------------------
def main():
    # Başlık ve Açıklama
    st.title("⚡ Elektrikli Araç Şarj Talebi ve Şebeke Yönetimi KDS")
    st.markdown(
        """
        **Aşama 1: B0 Noktasal Yük Tahmini ve S1 Belirsizlik Üst Sınırı**  
        Bu ekran, şebeke işletmecisi ve istasyon yöneticisi için ertesi günün (24 saatlik) güç talebini
        ve talep aşımlarını engellemek için kalibre edilmiş **q=0.90 güvenli üst sınırı** üretir.
        """
    )
    st.divider()

    # Yan Menü Kontrolleri
    with st.sidebar:
        st.header("⚙️ Model Parametreleri")
        secilen_senaryo = st.selectbox(
            "Yük Senaryosu Seçin:",
            options=list(SCENARIOS.keys()),
            index=0
        )
        st.markdown("---")
        st.markdown("### 🕒 Zaman Çerçevesi")
        st.info(
            f"**Origin (Tahmin Anı):**\n`31.08.2022 23:00`\n\n"
            f"**Hedef Dönem:**\n`01.09.2022 00:00 - 23:00` (24 Saat)\n\n"
            f"**Profil Tamponu (`BOSLUK`):**\n`2 Gün (Kesme: 30.08.2022)`\n\n"
            f"**Kalibrasyon:**\n`Son 8 Hafta (q=0.90)`"
        )
        st.markdown("---")
        st.caption("🔒 *Eylül 2022 test verisi eğitim ve kalibrasyona dahil edilmemiştir.*")

    # Tahminleri Üret
    try:
        df_tahmin, seviye, cal_info = tahmin_24h_uret(secilen_senaryo)
    except Exception as e:
        st.error(f"Tahmin üretilirken bir hata oluştu: {str(e)}")
        return

    # Temel Metrik Kartları
    col1, col2, col3, col4 = st.columns(4)
    
    b0_toplam_enerji = df_tahmin["B0_Tahmin_kW"].sum()
    b0_zirve = df_tahmin["B0_Tahmin_kW"].max()
    b0_zirve_saat = df_tahmin.loc[df_tahmin["B0_Tahmin_kW"].idxmax(), "Saat"]
    
    s1_zirve = df_tahmin["S1_Ust_Sinir_kW"].max()
    s1_zirve_saat = df_tahmin.loc[df_tahmin["S1_Ust_Sinir_kW"].idxmax(), "Saat"]
    
    with col1:
        st.metric(
            label="Toplam B0 Enerji",
            value=f"{b0_toplam_enerji:,.1f} kWh",
            help="01.09.2022 boyunca beklenen 24 saatlik toplam elektrik tüketimi."
        )
    with col2:
        st.metric(
            label="B0 Tepe Yük (Peak)",
            value=f"{b0_zirve:.1f} kW",
            delta=f"Saat {b0_zirve_saat}",
            delta_color="off",
            help="Noktasal tahminin ulaştığı en yüksek saatlik ortalama güç talebi."
        )
    with col3:
        st.metric(
            label="S1 Güvenli Üst Sınır (Peak)",
            value=f"{s1_zirve:.1f} kW",
            delta=f"Saat {s1_zirve_saat}",
            delta_color="off",
            help="q=0.90 kalibre edilmiş güvenli kapasite üst sınırının zirvesi."
        )
    with col4:
        st.metric(
            label="Origin Seviyesi",
            value=f"{seviye:.1f} kW",
            help="31.08.2022 23:00 anında bilinen son 168 saatlik şarj başlangıç enerjisi ortalaması."
        )

    st.markdown("### 📈 24 Saatlik Saatlik Güç Talebi ve Güvenli Üst Sınır")

    # İnteraktif Plotly Grafiği
    fig = go.Figure()

    # S1 Üst Sınır Bölgesi (Shaded Band)
    fig.add_trace(go.Scatter(
        x=df_tahmin["Saat"],
        y=df_tahmin["S1_Ust_Sinir_kW"],
        mode="lines",
        line=dict(color="rgba(255, 127, 14, 0.6)", width=1.5, dash="dash"),
        name="S1 Güvenli Üst Sınır (q=0.90)",
        hovertemplate="Saat %{x}<br>S1 Üst Sınır: %{y:.1f} kW<extra></extra>"
    ))

    # B0 Noktasal Tahmin Çizgisi
    fig.add_trace(go.Scatter(
        x=df_tahmin["Saat"],
        y=df_tahmin["B0_Tahmin_kW"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6, color="#1f77b4"),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.12)",
        name="B0 Noktasal Yük Tahmini",
        hovertemplate="Saat %{x}<br>B0 Tahmin: %{y:.1f} kW<extra></extra>"
    ))

    fig.update_layout(
        title=f"01.09.2022 Saatlik Şarj Yükü Tahmini — Senaryo: {secilen_senaryo}",
        xaxis_title="Günün Saati",
        yaxis_title="Ortalama Güç Talebi (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detaylı Tablo ve İndirme
    st.markdown("### 📋 Saatlik Tahmin Çizelgesi")
    
    tablo_gosterim = df_tahmin[["Saat", "B0_Tahmin_kW", "S1_Ust_Sinir_kW", "Guvenlik_Payi_kW"]].copy()
    tablo_gosterim.columns = ["Saat", "B0 Tahmin (kW)", "S1 Üst Sınır (kW)", "Güvenlik Payı (kW)"]
    
    col_tablo, col_indir = st.columns([3, 1])
    with col_tablo:
        st.dataframe(tablo_gosterim, use_container_width=True, height=300)
    with col_indir:
        st.markdown("**Veri İndirme**")
        st.caption("Aşağıdaki düğmeyi kullanarak 24 saatlik tahmin sonuçlarını CSV olarak kaydedebilirsiniz.")
        csv_veri = df_tahmin.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="📥 Tahminleri CSV İndir",
            data=csv_veri,
            file_name=f"tahmin_01_09_2022_{secilen_senaryo[:4].strip()}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Teşhis ve Doğrulama Paneli (Expander)
    with st.expander("🔍 Model Teşhisi, Kalibrasyon ve Metodoloji Doğrulaması", expanded=False):
        st.markdown(
            f"""
            #### 1. B0 Motoru Formülasyonu
            $$B_0(t) = \\frac{{\\text{{Profil}}(wd, hr)}}{{\\bar{{Y}}}} \\times \\text{{Seviye}}_{{\\text{{origin}}}}$$
            - **168 Hücrelik Profil Aralığı:** `07.10.2021 18:00` ile `{cal_info['kesme'].strftime('%d.%m.%Y %H:%M')}` arası.
            - **2 Günlük Tampon (`BOSLUK`):** Tahmin hedefi 01.09.2022 00:00 olduğundan, profil kesme tarihi `30.08.2022 00:00` olarak kilitlenmiştir.
            - **Origin Seviyesi:** 31.08.2022 23:00 itibarıyla bilinen son 168 saatin şarj başlangıç enerjisi ortalamasıdır (`{seviye:.2f} kWh`).

            #### 2. S1 Kalibrasyon Parametreleri (q=0.90)
            $$S_1(t) = \\max\\left(0, B_0(t) + z_{{0.90}} \\times \\sqrt{{B_0(t) + 0.05 \\times m}}\\right)$$
            - **Kalibrasyon Dönemi:** `{cal_info['cal_bas'].strftime('%d.%m.%Y')}` - `{cal_info['kesme'].strftime('%d.%m.%Y')}` (Son 8 hafta, {cal_info['n_cal']} tahmin kaydı).
            - **Ortalama Kalibrasyon Yükü ($m$):** `{cal_info['m']:.2f} kW`
            - **Karekök Ölçek Standartlaştırılmış Kalibrasyon Katsayısı ($z_{{0.90}}$):** `{cal_info['z_q']:.4f}`
            - **Artık Değer Aralığı ($z_{{\\min}} - z_{{\\max}}$):** `{cal_info['z_min']:.2f}` ile `{cal_info['z_max']:.2f}`

            #### 3. Veri Bütünlüğü ve Güvenlik Denetimleri
            - ✅ **Test Verisi İzolasyonu:** Eylül 2022 gerçekleşen değerleri B0 profilinde ve S1 kalibrasyonunda kesinlikle kullanılmamıştır.
            - ✅ **Kalibrasyon Profil İzolasyonu:** Kalibrasyon dönemi tahminleri üretilirken kullanılan profil, kalibrasyon başlangıç tarihinden (`{cal_info['cal_bas'].strftime('%d.%m.%Y')}`) önceki geçmiş seriden hesaplanmıştır (bilgi sızıntısı engellenmiştir).
            - ✅ **Fiziksel Tutarlılık:** Negatif yük tahmini ve negatif üst sınır oluşumu engellenmiştir (`clip(0, None)`).
            """
        )

if __name__ == "__main__":
    main()
