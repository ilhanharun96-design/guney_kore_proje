# -*- coding: utf-8 -*-
"""
Elektrikli Araç Şarj Talebi ve Şebeke Kapasitesi Karar Destek Sistemi (KDS)
Aşama 1: B0 Noktasal Yük Tahmini ve S1 Belirsizlik Üst Sınırı Motoru

Origin: 31.08.2022 23:00
Hedef: 01.09.2022 00:00 - 23:00 (24 saatlik ufuk)
Metodoloji: 168 hücrelik (gün x saat) B0 profil ölçekleme, 2 günlük BOSLUK tamponu,
            ve son 8 haftalık kalibrasyon artıklarıyla hesaplanan q=0.90 S1 karekök üst sınırı.
"""

# SONRA
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import simulasyon_motoru as sm

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
    "F50 (Enerjinin İlk %50'sinde Verilmesi Varsayımı)": "yuk_F50_kwh",
    "F25 (Enerjinin İlk %25'inde Verilmesi Varsayımı)": "yuk_F25_kwh"
}

P99_REFS = {
    "U100": 329.8,
    "F50": 386.2,
    "F25": 429.7
}
SC_SUMMARY_PATH = BASE_DIR / "outputs_smart_charging" / "akilli_sarj_ozet_tablo.csv"

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


@st.cache_data
def akilli_sarj_ozet_yukle():
    """outputs_smart_charging/akilli_sarj_ozet_tablo.csv dosyasını güvenle yükler."""
    if SC_SUMMARY_PATH.exists():
        return pd.read_csv(SC_SUMMARY_PATH)
    return None

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
# Aşama 2: Kapasite Analizi ve Karar Destek Ekranı
# -----------------------------------------------------------------------------
def render_asama2_kapasite_analizi(df_tahmin: pd.DataFrame, sc_key: str):
    """
    Aşama 2: Kapasite Analizi ve Operasyonel Karar Destek Ekranı.
    Aşama 1 tarafından üretilen df_tahmin verisini yalnızca okur; B0 ve S1 değerlerini yeniden hesaplamaz.
    """
    st.markdown("### ⚡ Kapasite Sınırı ve Aşım Analizi")
    st.markdown(
        """
        Bu ekran, operatörün belirleyeceği şebeke / trafo güç kapasitesi sınırı altında
        ertesi günün (01.09.2022) beklenen yük talebini ve ihtiyatlı belirsizlik riskini değerlendirir.
        """
    )

    # Senaryo kısa adı ve referans p99 kapasitesi
    sc_short = "U100" if "U100" in sc_key else ("F50" if "F50" in sc_key else "F25")
    ref_p99 = P99_REFS[sc_short]
    p99_str = f"{ref_p99:.1f}".replace(".", ",")

    # Kapasite belirleme kontrolleri (Slider ve Sayısal Giriş senkronizasyonu)
    st.markdown("#### 🎛️ Kapasite Sınırı Belirleme")
    st.caption(
        f"Seçili Senaryo ({sc_short}) için eğitim dönemi %99 tepe yük referansı: {p99_str} kW"
    )

    slider_key = f"slider_pct_{sc_short}"
    num_key = f"num_kw_{sc_short}"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = 80.0
    if num_key not in st.session_state:
        st.session_state[num_key] = float(round(ref_p99 * 0.80, 1))

    def _sync_from_slider():
        pct = st.session_state[slider_key]
        st.session_state[num_key] = float(round(ref_p99 * (pct / 100.0), 1))

    def _sync_from_num():
        val = max(0.0, float(st.session_state[num_key]))
        st.session_state[num_key] = val
        pct = round((val / ref_p99) * 100.0, 1)
        st.session_state[slider_key] = float(min(110.0, max(60.0, pct)))

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.slider(
            "Kapasite Oranı (% p99 Referansı)",
            min_value=60.0,
            max_value=110.0,
            step=1.0,
            key=slider_key,
            on_change=_sync_from_slider,
            help=f"Eğitim dönemi %99 tepe yüküne ({ref_p99:.1f} kW) oranla kapasite sınırını belirler."
        )
    with col_k2:
        st.number_input(
            "Kapasite Sınırı $C_{\\text{limit}}$ (kW)",
            min_value=0.0,
            step=5.0,
            key=num_key,
            on_change=_sync_from_num,
            help="Şebeke veya trafo için izin verilen azami saatlik ortalama güç sınırı (kW)."
        )

    c_limit = float(st.session_state[num_key])
    kap_orani = (c_limit / ref_p99) * 100.0

    # Saatlik hesaplamalar (df_tahmin yalnızca okunur)
    b0_vals = df_tahmin["B0_Tahmin_kW"].to_numpy()
    s1_vals = df_tahmin["S1_Ust_Sinir_kW"].to_numpy()

    b0_asim_gucu = np.maximum(0.0, b0_vals - c_limit)
    s1_asim_gucu = np.maximum(0.0, s1_vals - c_limit)

    # 1 saatlik aralıkta enerji = güç * 1 saat
    b0_asim_enerji = b0_asim_gucu * 1.0  # kWh
    s1_asim_enerji = s1_asim_gucu * 1.0  # kWh

    b0_asim_saat_sayisi = int((b0_asim_gucu > 0.01).sum())
    s1_asim_saat_sayisi = int((s1_asim_gucu > 0.01).sum())

    toplam_b0_asim_enerji = float(b0_asim_enerji.sum())
    toplam_s1_asim_enerji = float(s1_asim_enerji.sum())

    # Durum sınıflandırması
    durumlar = []
    for b, s in zip(b0_vals, s1_vals):
        if s <= c_limit:
            durumlar.append("S1 sınırı altında")
        elif b <= c_limit:
            durumlar.append("İhtiyatlı üst sınır aşımı (Belirsizlik riski)")
        else:
            durumlar.append("Noktasal tahmine göre kapasite aşımı")

    st.markdown("---")

    # Dört Temel KPI Göstergesi
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Kullanıcı Kapasite Sınırı",
            value=f"{c_limit:.1f} kW",
            delta=f"%{kap_orani:.1f} p99 ref",
            delta_color="off",
            help="Operatör tarafından belirlenen şebeke güç kapasitesi."
        )
    with kpi2:
        st.metric(
            label="B0 Aşım Saati",
            value=f"{b0_asim_saat_sayisi} saat",
            delta=f"{toplam_b0_asim_enerji:.1f} kWh aşım",
            delta_color="inverse" if b0_asim_saat_sayisi > 0 else "off",
            help="Noktasal tahmine göre kapasite sınırının aşıldığı saat sayısı ve toplam aşım enerjisi."
        )
    with kpi3:
        st.metric(
            label="S1 Risk Saati",
            value=f"{s1_asim_saat_sayisi} saat",
            delta=f"{toplam_s1_asim_enerji:.1f} kWh risk",
            delta_color="inverse" if s1_asim_saat_sayisi > 0 else "off",
            help="q=0.90 ihtiyatlı üst sınırın kapasiteyi aştığı saat sayısı ve bu aşımın toplam enerji karşılığı."
        )
    with kpi4:
        if b0_asim_saat_sayisi > 0:
            genel_durum_text = "Noktasal Aşım Bekleniyor"
            durum_delta = "Müdahale Gerekli"
            durum_color = "inverse"
        elif s1_asim_saat_sayisi > 0:
            genel_durum_text = "İhtiyatlı Risk Mevcut"
            durum_delta = "İzleme Önerilir"
            durum_color = "off"
        else:
            genel_durum_text = "S1 Sınırı Altında"
            durum_delta = None
            durum_color = "off"
        st.metric(
            label="Operasyonel Durum",
            value=genel_durum_text,
            delta=durum_delta,
            delta_color=durum_color,
            help="24 saatlik ufuktaki genel kapasite yeterlilik özeti."
        )

    # Grafikler
    st.markdown("#### 📊 Yük, Kapasite ve Aşım Grafikleri")

    # Grafik 1: Talep Eğrisi ve Kapasite Sınırı
    fig_cap = go.Figure()

    # S1 Üst Sınır
    fig_cap.add_trace(go.Scatter(
        x=df_tahmin["Saat"],
        y=df_tahmin["S1_Ust_Sinir_kW"],
        mode="lines",
        line=dict(color="rgba(255, 127, 14, 0.7)", width=1.5, dash="dash"),
        name="S1 İhtiyatlı Üst Sınır (q=0.90)",
        hovertemplate="Saat %{x}<br>S1: %{y:.1f} kW<extra></extra>"
    ))

    # B0 Noktasal Tahmin
    fig_cap.add_trace(go.Scatter(
        x=df_tahmin["Saat"],
        y=df_tahmin["B0_Tahmin_kW"],
        mode="lines+markers",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6, color="#1f77b4"),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.12)",
        name="B0 Noktasal Yük Tahmini",
        hovertemplate="Saat %{x}<br>B0: %{y:.1f} kW<extra></extra>"
    ))

    # Kapasite Sınırı Çizgisi
    fig_cap.add_hline(
        y=c_limit,
        line_dash="dot",
        line_color="#d62728",
        line_width=2.5,
        annotation_text=f"Kapasite Sınırı: {c_limit:.1f} kW",
        annotation_position="top left",
        annotation_font_color="#d62728"
    )

    fig_cap.update_layout(
        title=f"01.09.2022 Yük Talebi ve Kapasite Sınırı Karşılaştırması — {sc_short}",
        xaxis_title="Günün Saati",
        yaxis_title="Güç Talebi / Kapasite (kW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
        template="plotly_white"
    )
    st.plotly_chart(fig_cap, use_container_width=True)

    # Grafik 2: Saatlik Aşım Gücü Sütun Grafiği
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_tahmin["Saat"],
        y=np.round(b0_asim_gucu, 2),
        name="B0 Aşım Gücü (kW)",
        marker_color="#d62728",
        hovertemplate="Saat %{x}<br>B0 Aşım Gücü: %{y:.2f} kW<br>Aşım Enerjisi: %{y:.2f} kWh<extra></extra>"
    ))
    fig_bar.add_trace(go.Bar(
        x=df_tahmin["Saat"],
        y=np.round(np.maximum(0.0, s1_asim_gucu - b0_asim_gucu), 2),
        name="İhtiyatlı S1 Ek Risk Payı (kW)",
        marker_color="#ff7f0e",
        hovertemplate="Saat %{x}<br>S1 Ek Risk Payı: %{y:.2f} kW<extra></extra>"
    ))
    fig_bar.update_layout(
        title=f"Saatlik Kapasite Aşım ve Belirsizlik Riski Gücü (kW) — {sc_short}",
        xaxis_title="Günün Saati",
        yaxis_title="Kapasiteyi Aşan Güç (kW)",
        barmode="stack",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=320,
        template="plotly_white"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Saatlik Detay Tablosu
    st.markdown("#### 📋 Saatlik Kapasite ve Aşım Detay Tablosu")

    df_kap_detay = pd.DataFrame({
        "Saat": df_tahmin["Saat"],
        "B0 Tahmini (kW)": df_tahmin["B0_Tahmin_kW"],
        "S1 Üst Sınırı (kW)": df_tahmin["S1_Ust_Sinir_kW"],
        "Kapasite Sınırı (kW)": np.round(c_limit, 2),
        "B0 Aşım Gücü (kW)": np.round(b0_asim_gucu, 2),
        "S1 Aşım Gücü (kW)": np.round(s1_asim_gucu, 2),
        "B0 Aşım Enerjisi (kWh)": np.round(b0_asim_enerji, 2),
        "S1 Aşım Enerjisi (kWh)": np.round(s1_asim_enerji, 2),
        "Durum": durumlar
    })

    st.dataframe(df_kap_detay, use_container_width=True, height=350)

    # Tarihsel Referans: Ağustos 2022 Akıllı Şarj Simülasyon Deneyleri
    with st.expander("📚 Tarihsel Referans: Ağustos 2022 Akıllı Şarj Simülasyon Deneyleri", expanded=False):
        st.markdown(
            """
            > [!IMPORTANT]
            > **Tarihsel Deney Notu:**  
            > Aşağıdaki tablo **01.09.2022 günü için üretilmiş anlık bir tahmin sonucu değildir.**  
            > Ağustos 2022 değerlendirme döneminde 8.016 şarj oturumu ve 15 dakikalık zaman adımlarıyla
            > çalıştırılan kural tabanlı (LLF - Least Laxity First) tarihsel akıllı şarj simülasyonlarının
            > sonuçlarını göstermektedir. Sonuçlar, belirli kapasite ve esneklik varsayımları altında
            > kural tabanlı şarj yönetiminin karşılanamayan enerji miktarına etkisini incelemek amacıyla
            > kullanılmıştır. Gerçek bir şebeke uygulamasının performans garantisi değildir.
            """
        )
        sim_df = akilli_sarj_ozet_yukle()
        if sim_df is not None:
            sim_sc = sim_df[sim_df["senaryo"] == sc_short].copy()
            if not sim_sc.empty:
                gosterim_kolonlari = {
                    "kapasite_orani": "Kapasite Oranı",
                    "kapasite_kw": "Kapasite (kW)",
                    "esnek_pay": "Esnek Payı",
                    "bekleme_saat": "Ek Bekleme (sa)",
                    "kontrolsuz_asim_saat": "Kontrolsüz Aşım (sa)",
                    "kontrolsuz_asim_enerji_pct": "Kontrolsüz Aşım Enerjisi (%)",
                    "karsilanamayan_pct": "Karşılanamayan Enerji (%)",
                    "ertelenen_pct": "Ertelenen Enerji (%)",
                    "ort_gecikme_dk": "Ortalama Gecikme (dk)"
                }
                mevcut_kol = [k for k in gosterim_kolonlari.keys() if k in sim_sc.columns]
                sim_goster = sim_sc[mevcut_kol].rename(columns=gosterim_kolonlari)
                st.dataframe(sim_goster.round(2), use_container_width=True)
            else:
                st.info(f"{sc_short} senaryosu için simülasyon kaydı bulunamadı.")
        else:
            st.warning("`outputs_smart_charging/akilli_sarj_ozet_tablo.csv` dosyası bulunamadı.")

    # Bilimsel Sınırlamalar
    with st.expander("⚠️ Bilimsel ve Metodolojik Sınırlamalar", expanded=False):
        st.markdown(
            """
            - **Gerçek Şebeke Kapasitesi:** Veri setinde trafo gücü veya şebeke bağlantı sınırı bulunmamaktadır; kapasite sınırı kullanıcı tarafından belirlenen bir senaryo parametresidir.
            - **Şebeke Kontrolü ve Optimizasyon:** Sistem gerçek zamanlı şebeke kontrolü veya araç bazında optimum şarj planlaması yapmaz; ağ düzeyinde analitik bir karar destek aracıdır.
            - **Tahmin ve Belirsizlik Doğası:** B0 aşımı kesin gerçekleşecek bir olay değil, noktasal tahmine göre beklenen değerdir. S1 üst sınırı saat bazında kesin bir olasılık garantisi olmayıp, $q=0.90$ kalibrasyonlu ihtiyatlı üst sınırdır.
            - **Günlük Aşım Şiddeti:** Günlük aşım şiddeti sıralaması deneyi pre-registered $S_2$ asgari güvenilirlik eşiğini ($r_s \\ge 0.60$) geçemediği için ekranda otomatik şiddet sıralaması veya alarmı yer almamaktadır.
            - **F50 ve F25 Senaryoları:** Ölçülmüş gerçek değerler olmayıp, enerjinin oturum süresinin bir kısmında verildiğini varsayan yük dağıtım senaryolarıdır.
            """
        )

# -----------------------------------------------------------------------------
# Arayüz Bileşenleri
# -----------------------------------------------------------------------------
def main():
    # Başlık ve Açıklama
    st.title("⚡ Elektrikli Araç Şarj Talebi ve Şebeke Yönetimi KDS")
    st.markdown(
        """
        **Karar Destek Sistemi (KDS)** — Ertesi gün (24 saatlik) güç talebi tahmini,
        $q=0.90$ kalibre edilmiş ihtiyatlı belirsizlik üst sınırı ve şebeke kapasite analizi.
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

    # Sekmeler: Aşama 1 ve Aşama 2
    # SONRA
    tab1, tab2, tab3 = st.tabs([
        "📈 Aşama 1: B0 & S1 Yük Tahmini",
        "⚡ Aşama 2: Yük ve Kapasite Analizi",
        "🔌 Aşama 3: Akıllı Şarj Senaryoları"
    ])

    with tab1:
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
                label="S1 İhtiyatlı Üst Sınır (Peak)",
                value=f"{s1_zirve:.1f} kW",
                delta=f"Saat {s1_zirve_saat}",
                delta_color="off",
                help="q=0.90 kalibrasyonlu ihtiyatlı üst sınırın ulaştığı en yüksek saatlik güç değeri."
            )
        with col4:
            st.metric(
                label="Origin Seviyesi",
                value=f"{seviye:.1f} kW",
                help="31.08.2022 23:00 anında bilinen son 168 saatlik şarj başlangıç enerjisi ortalaması."
            )

        st.markdown("### 📈 24 Saatlik Güç Talebi ve İhtiyatlı Üst Sınır")

        # İnteraktif Plotly Grafiği
        fig = go.Figure()

        # S1 Üst Sınır Bölgesi (Shaded Band)
        fig.add_trace(go.Scatter(
            x=df_tahmin["Saat"],
            y=df_tahmin["S1_Ust_Sinir_kW"],
            mode="lines",
            line=dict(color="rgba(255, 127, 14, 0.6)", width=1.5, dash="dash"),
            name="S1 İhtiyatlı Üst Sınır (q=0.90)",
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
        tablo_gosterim.columns = ["Saat", "B0 Tahmin (kW)", "S1 Üst Sınır (kW)", "Belirsizlik Payı (kW)"]

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

    # SONRA
    with tab2:
        render_asama2_kapasite_analizi(df_tahmin, secilen_senaryo)

    with tab3:
        st.markdown("### Akıllı Şarj Senaryoları")
        st.markdown(
            "Farklı kapasite ve esneklik varsayımları altında akıllı şarj "
            "simülasyonlarının karşılaştırılması."
        )
        st.caption(
            "Bu ekran duyarlılık analizi niteliğindedir; gerçek zamanlı kontrol "
            "sistemi veya gerçek şebeke performans garantisi değildir."
        )
        st.divider()

        # --- 1) Bağımsız senaryo seçici (Aşama 3'e özel) ---
        if "a3_senaryo" not in st.session_state:
            st.session_state["a3_senaryo"] = secilen_senaryo

        a3_senaryo = st.selectbox(
            "Yük Dağılım Senaryosu",
            options=list(SCENARIOS.keys()),
            key="a3_senaryo",
            format_func=lambda k: k.split(" ")[0],
            help="Bu seçim yalnızca Aşama 3 simülasyon parametrelerini etkiler.",
        )
        sc_short_a3 = (
            "U100" if "U100" in a3_senaryo
            else ("F50" if "F50" in a3_senaryo else "F25")
        )

        # --- Dosya imzası + hassas P99 ---
        try:
            imza_a3 = sm.dosya_imzasi()
            ref_kw_p99_hassas = sm.ref_kw_hesapla(sc_short_a3, imza_a3)
        except Exception as _e:
            st.error(f"Referans P99 hesaplanamadı: {_e}")
            st.stop()

        _a3_min_kw = float(round(ref_kw_p99_hassas * 0.60, 3))
        _a3_max_kw = float(round(ref_kw_p99_hassas * 1.10, 3))

        # --- 2) Kapasite sınırı ---
        st.markdown("#### Kapasite Sınırı")
        st.caption(
            f"P99 referansı ({sc_short_a3}): **{ref_kw_p99_hassas:.3f} kW** — "
            f"eğitim dönemindeki (07.10.2021 – 31.07.2022) saatlik yüklerin "
            f"%99'luk dilim değeri. Gösterim için yuvarlanmış: "
            f"≈ {sm.P99_BEKLENEN_KW[sc_short_a3]:.1f} kW. "
            f"Simülasyon kapasitesi bu hassas değer üzerinden hesaplanır."
        )

        _a3_slider_key = f"a3_slider_pct_{sc_short_a3}"
        _a3_num_key = f"a3_num_kw_{sc_short_a3}"

        if _a3_slider_key not in st.session_state:
            st.session_state[_a3_slider_key] = 80.0
        if _a3_num_key not in st.session_state:
            st.session_state[_a3_num_key] = float(
                round(ref_kw_p99_hassas * 0.80, 3)
            )

        def _a3_sync_from_slider():
            pct = st.session_state[_a3_slider_key]
            st.session_state[_a3_num_key] = float(
                round(ref_kw_p99_hassas * (pct / 100.0), 3)
            )

        def _a3_sync_from_num():
            val = float(st.session_state[_a3_num_key])
            val = max(_a3_min_kw, min(_a3_max_kw, val))
            pct_raw = (val / ref_kw_p99_hassas) * 100.0
            pct_snapped = round(pct_raw / 5.0) * 5.0
            pct_snapped = max(60.0, min(110.0, pct_snapped))
            kw_snapped = float(round(ref_kw_p99_hassas * (pct_snapped / 100.0), 3))
            st.session_state[_a3_slider_key] = float(pct_snapped)
            st.session_state[_a3_num_key] = kw_snapped

        _a3_col1, _a3_col2 = st.columns(2)
        with _a3_col1:
            st.slider(
                "Kapasite Oranı (%P99)",
                min_value=60.0,
                max_value=110.0,
                step=5.0,
                key=_a3_slider_key,
                on_change=_a3_sync_from_slider,
                help="Eğitim dönemi P99 referansına oran.",
            )
        with _a3_col2:
            st.number_input(
                "Kapasite Sınırı (kW)",
                min_value=_a3_min_kw,
                max_value=_a3_max_kw,
                step=0.001,
                format="%.3f",
                key=_a3_num_key,
                on_change=_a3_sync_from_num,
                help="Saatlik ortalama güç sınırı (kW). %60–%110 P99 aralığında. "
                     "Slider ile iki yönlü senkronize.",
            )

        # --- 3-4) Esneklik parametreleri ---
        st.markdown("#### Esneklik Parametreleri")
        _a3_col3, _a3_col4 = st.columns(2)
        with _a3_col3:
            esnek_pay_a3 = st.selectbox(
                "Esnek Kullanıcı Oranı",
                options=[0.0, 0.25, 0.50],
                index=1,
                format_func=lambda x: f"%{int(x * 100)}",
                key="a3_esnek_pay",
                help="Şarjını esnetmeye gönüllü kullanıcıların oranı (rastgele seçim).",
            )
        with _a3_col4:
            bekleme_saat_a3 = st.selectbox(
                "Ek Bekleme Süresi",
                options=[0, 1, 2],
                index=2,
                format_func=lambda x: f"{x} saat",
                key="a3_bekleme_saat",
                help="Esnek kullanıcıların oturumuna eklenebilecek azami bekleme.",
            )

        # --- 5) Hesapla butonu ---
        st.divider()
        _a3_hesapla = st.button(
            "Hesapla", key="a3_hesapla_btn", use_container_width=True
        )
        if _a3_hesapla:
            _a3_tekrar_on = 1 if esnek_pay_a3 == 0 else 5
            with st.spinner(
                f"Simülasyon çalışıyor ({_a3_tekrar_on} tekrar)..."
            ):
                try:
                    _a3_sonuc_yeni = sm.calistir_simulasyon(
                        sc_short_a3,
                        st.session_state[_a3_slider_key] / 100.0,
                        esnek_pay_a3,
                        float(bekleme_saat_a3),
                        imza_a3,
                    )
                    st.session_state["a3_sonuc"] = _a3_sonuc_yeni
                    st.success("Simülasyon tamamlandı.")
                except Exception as _e:
                    st.session_state.pop("a3_sonuc", None)
                    st.error(f"Simülasyon hatası: {_e}")

        # --- 6) Sonuç görüntüleme ---
        if "a3_sonuc" in st.session_state:
            _sonuc = st.session_state["a3_sonuc"]
            _meta = _sonuc["meta"]

            _kapasite_guncel = st.session_state[_a3_slider_key] / 100.0
            _params_guncel = (
                _meta["senaryo"] == sc_short_a3
                and abs(_meta["kapasite_orani_p99"] - _kapasite_guncel) < 1e-9
                and abs(_meta["esnek_pay"] - esnek_pay_a3) < 1e-9
                and abs(_meta["bekleme_saat"] - float(bekleme_saat_a3)) < 1e-9
                and tuple(_meta["dosya_imzasi"]) == tuple(imza_a3)
            )
            if not _params_guncel:
                st.warning(
                    "Parametreler (veya kaynak veri) değişti. Aşağıdaki sonuçlar "
                    "**önceki parametrelerle** hesaplanmıştır. Güncel sonuçlar için "
                    "Hesapla'ya basın."
                )

            # Standart sapma etiketi: tek tekrarda sapma tanımsızdır.
            def _a3_std_etiketi(tekrar_efektif, std_deger):
                if tekrar_efektif <= 1:
                    return "± — (tek tekrar)"
                return f"± {std_deger:.2f}"

            _tekrar_ef = int(_meta["tekrar_efektif"])

            st.divider()
            st.markdown("#### Simülasyon Bilgileri")
            _b1, _b2, _b3 = st.columns(3)
            with _b1:
                st.metric("Senaryo", _meta["senaryo"])
                st.metric("Kapasite Sınırı (kW)", f"{_meta['kapasite_kw']:.3f}")
            with _b2:
                st.metric(
                    "Kapasite Oranı (%P99)",
                    f"%{_meta['kapasite_orani_p99']*100:.0f}",
                )
                st.metric(
                    "Esnek Kullanıcı Oranı", f"%{int(_meta['esnek_pay']*100)}"
                )
            with _b3:
                st.metric(
                    "Ek Bekleme Süresi", f"{_meta['bekleme_saat']:.0f} saat"
                )
                st.metric("Tekrar Sayısı", str(_tekrar_ef))
            st.caption(
                f"Değerlendirme dönemi: **{_meta['degerlendirme_bas']} – "
                f"{_meta['degerlendirme_bit']}** (Ağustos 2022 tarihsel verisi). "
                f"Toplam değerlendirme oturumu: **{_meta['oturum_sayisi']}**. "
                f"Toplam enerji: **{_meta['toplam_enerji_kwh']:.1f} kWh**."
            )

            st.markdown("#### Kontrolsüz Yük (Kapasite Sınırı Yokmuş Gibi)")
            st.caption(
                "Kontrolsüz yük deterministiktir; tekrar sayısından bağımsızdır."
            )
            _k1, _k2, _k3 = st.columns(3)
            with _k1:
                st.metric(
                    "Kapasite üstünde kalınan süre (saat)",
                    f"{_sonuc['kontrolsuz']['asim_sure_saat']:.2f}",
                    help="Kapasite sınırının aşıldığı 15 dakikalık aralık "
                         "sayısı × 0.25 saat.",
                )
            with _k2:
                st.metric(
                    "Kapasite üstü enerji (kWh)",
                    f"{_sonuc['kontrolsuz']['asim_enerji_kwh']:.2f}",
                    help="Kapasite sınırının üzerinde kalan toplam enerji.",
                )
            with _k3:
                st.metric(
                    "Kapasite üstü enerji oranı (%)",
                    f"{_sonuc['kontrolsuz']['asim_enerji_pct']:.2f}",
                    help="Kapasite üstü enerji / toplam kontrolsüz enerji × 100.",
                )

            st.markdown("#### Akıllı Şarj Sonucu")
            if _tekrar_ef <= 1:
                st.caption(
                    "Tek tekrar çalıştırıldı (esnek kullanıcı oranı %0). "
                    "Standart sapma tanımsızdır."
                )
            else:
                st.caption(
                    f"Değerler {_tekrar_ef} tekrar üzerinden ortalamadır; "
                    f"± değerler örneklem standart sapmasıdır (ddof=1)."
                )
            _a1, _a2, _a3c, _a4, _a5 = st.columns(5)
            with _a1:
                st.metric(
                    "Karşılanamayan Enerji (%)",
                    f"{_sonuc['akilli']['karsilanamayan_pct_ortalama']:.2f}",
                    delta=_a3_std_etiketi(
                        _tekrar_ef,
                        _sonuc["akilli"]["karsilanamayan_pct_std"],
                    ),
                    delta_color="off",
                )
            with _a2:
                st.metric(
                    "Ertelenen Enerji (%)",
                    f"{_sonuc['akilli']['ertelenen_pct_ortalama']:.2f}",
                    delta=_a3_std_etiketi(
                        _tekrar_ef,
                        _sonuc["akilli"]["ertelenen_pct_std"],
                    ),
                    delta_color="off",
                )
            with _a3c:
                st.metric(
                    "Geciken Oturum (%)",
                    f"{_sonuc['akilli']['geciken_oturum_pct_ortalama']:.2f}",
                    delta=_a3_std_etiketi(
                        _tekrar_ef,
                        _sonuc["akilli"]["geciken_oturum_pct_std"],
                    ),
                    delta_color="off",
                )
            with _a4:
                st.metric(
                    "Ortalama Gecikme (dk)",
                    f"{_sonuc['akilli']['ort_gecikme_dk_ortalama']:.2f}",
                    delta=_a3_std_etiketi(
                        _tekrar_ef,
                        _sonuc["akilli"]["ort_gecikme_dk_std"],
                    ),
                    delta_color="off",
                )
            with _a5:
                st.metric(
                    "Ortalama Kontrollü Tepe Güç (kW)",
                    f"{_sonuc['akilli']['tepe_kw_ortalama']:.2f}",
                    delta=_a3_std_etiketi(
                        _tekrar_ef,
                        _sonuc["akilli"]["tepe_kw_std"],
                    ),
                    delta_color="off",
                )
            st.markdown(
                "#### Kontrolsüz ve Akıllı Şarj Sonrası Saatlik Yük Karşılaştırması"
            )
            _a3_x = pd.date_range(
                _meta["degerlendirme_bas"],
                _meta["degerlendirme_bit"],
                freq="h",
                inclusive="left",
            )
            _a3_y_kontrolsuz = np.asarray(
                _sonuc["kontrolsuz"]["saatlik_kw"], dtype=float
            )
            _a3_y_kontrollu = np.asarray(
                _sonuc["akilli"]["saatlik_kw_ortalama"], dtype=float
            )
            _a3_n = len(_a3_x)
            if not (
                _a3_y_kontrolsuz.shape == (_a3_n,)
                and _a3_y_kontrollu.shape == (_a3_n,)
            ):
                st.error(
                    "Grafik çizilemedi: saatlik yük dizileri ile tarih aralığı "
                    f"uzunluğu uyuşmuyor. Tarih: {_a3_n}, "
                    f"kontrolsüz: {_a3_y_kontrolsuz.size}, "
                    f"kontrollü: {_a3_y_kontrollu.size}."
                )
            else:
                _a3_fig = go.Figure()
                _a3_fig.add_trace(
                    go.Scatter(
                        x=_a3_x,
                        y=_a3_y_kontrolsuz,
                        mode="lines",
                        name="Kontrolsüz Yük (kW)",
                        line=dict(color="#1f77b4", width=1.6),
                        hovertemplate=(
                            "%{x|%d.%m %H:%M}"
                            "<br>Kontrolsüz: %{y:.1f} kW<extra></extra>"
                        ),
                    )
                )
                _a3_fig.add_trace(
                    go.Scatter(
                        x=_a3_x,
                        y=_a3_y_kontrollu,
                        mode="lines",
                        name="Akıllı Şarj Sonrası Yük (kW)",
                        line=dict(color="#2ca02c", width=1.6),
                        hovertemplate=(
                            "%{x|%d.%m %H:%M}"
                            "<br>Kontrollü: %{y:.1f} kW<extra></extra>"
                        ),
                    )
                )
                _a3_fig.add_hline(
                    y=_meta["kapasite_kw"],
                    line_dash="dot",
                    line_color="#d62728",
                    line_width=2.0,
                    annotation_text=(
                        f"Kapasite Sınırı: {_meta['kapasite_kw']:.3f} kW "
                        f"(%{_meta['kapasite_orani_p99']*100:.0f} P99)"
                    ),
                    annotation_position="top left",
                    annotation_font_color="#d62728",
                )
                _a3_fig.update_layout(
                    title=(
                        "Kontrolsüz ve Akıllı Şarj Sonrası Saatlik Yük "
                        f"Karşılaştırması — {_meta['senaryo']}, "
                        f"%{_meta['kapasite_orani_p99']*100:.0f} P99"
                    ),
                    xaxis_title="Tarih",
                    yaxis_title="Saatlik Ortalama Yük (kW)",
                    hovermode="x unified",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                    ),
                    margin=dict(l=40, r=40, t=60, b=40),
                    height=420,
                    template="plotly_white",
                )
                st.plotly_chart(_a3_fig, width="stretch")
                st.caption(
                    "Kontrollü yük serisi, simülasyondaki tekrarların saatlik "
                    "ortalamasıdır. Tekrarlar arası saatlik değişkenlik verisi "
                    "saklanmadığından grafik üzerinde belirsizlik bandı "
                    "gösterilmemektedir."
                )
            with st.expander(
                "⚠️ Simülasyon Hakkında Bilimsel Notlar", expanded=False
            ):
                st.markdown(
                    """
                    - Bu simülasyon **Ağustos 2022 tarihsel verileri** üzerinde çalışır; \
gerçek zamanlı bir kontrol sistemi değildir ve gelecek tahmini yapmaz.
                    - Algoritma (LLF - Least Laxity First) her oturumun **ayrılış \
zamanını** ve **enerji ihtiyacını** önceden bilir. Bu, simülasyonun **iyimser \
varsayımıdır**; gerçek operasyonda bu bilgiler belirsiz veya eksiktir.
                    - **U100** enerjinin oturum süresi boyunca sabit güçle verildiği \
temel varsayımdır. **F50** ve **F25** ise enerjinin oturumun ilk %50 / %25'inde \
verildiği **varsayımsal** duyarlılık senaryolarıdır; ölçülmüş gerçek değerler \
değildir.
                    - Sonuçlar gerçek şebeke performansı garantisi değildir. \
Kapasite referansı eğitim dönemi P99 değeri üzerinden türetilmiştir; gerçek \
trafo/şebeke kapasitesi verisi projede yoktur.
                    - Kapasite sınırı saatlik ortalama güç (kW) cinsindendir; \
kapasite üstü enerji kWh, kapasite üstünde kalınan süre saat olarak raporlanır.
                    - Bu ekran **duyarlılık analizi** niteliğindedir; otomatik \
uyarı veya karar motoru değildir.
                    """
                )
if __name__ == "__main__":
    main()