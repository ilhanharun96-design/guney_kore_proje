# -*- coding: utf-8 -*-
"""
Elektrikli Araç Şarj İstasyonlarında Saatlik Talep Tahmini
B0 Noktasal Tahmin ve S1 Belirsizlik Üst Sınırı

Uygulama bölümleri:
1. Oturum Başlangıç Tahmini
2. Başlangıç Enerjisi Tahmini
3. Yaklaşık Saatlik Yük Tahmini
4. Model Performansı ve Karşılaştırma
5. Talep Yoğunluğu Analizi

Tahmin tarihi: 01.09.2022
Tarihsel karar destek sistemi prototipidir.
Gerçek zamanlı şebeke kontrolü yapmaz.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. GENEL AYARLAR
# ============================================================

st.set_page_config(
    page_title="Elektrikli Araç Şarj Talebi KDS",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

FE_DIR = BASE_DIR / "outputs_feature_engineering"
LOAD_PATH = FE_DIR.parent / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"

TRAIN_PATH = FE_DIR / "train_verisi.csv"
VAL_PATH = FE_DIR / "validation_verisi.csv"

TEST_START = pd.Timestamp("2022-09-01 00:00:00")
FORECAST_HOURS = 24

PROFILE_START = pd.Timestamp("2021-10-07 18:00:00")

Q_LEVEL = 0.90
CALIBRATION_WEEKS = 8

ENERGY_LEVEL_COL = "started_session_energy_kwh_rolling_mean_168"

WEEKDAY_NAMES = [
    "Pazartesi",
    "Salı",
    "Çarşamba",
    "Perşembe",
    "Cuma",
    "Cumartesi",
    "Pazar",
]

LOAD_SCENARIOS = {
    "U100": "yuk_U100_kwh",
    "F50": "yuk_F50_kwh",
    "F25": "yuk_F25_kwh",
}


# ============================================================
# 2. VERİLERİ YÜKLEME
# ============================================================

@st.cache_data
def load_data():

    gerekli_dosyalar = [
        TRAIN_PATH,
        VAL_PATH,
        LOAD_PATH,
    ]

    for dosya in gerekli_dosyalar:
        if not dosya.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {dosya}"
            )

    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["target_datetime", "origin_datetime"]
    )

    validation = pd.read_csv(
        VAL_PATH,
        parse_dates=["target_datetime", "origin_datetime"]
    )

    tum = pd.concat(
        [train, validation],
        ignore_index=True
    )

    # Eylül 2022 test dönemi uygulama tahmin verisine alınmaz.
    tum = tum[
        tum["target_datetime"] < TEST_START
    ].copy()

    tum = tum.sort_values(
        ["target_datetime", "forecast_horizon"]
    ).reset_index(drop=True)

    # Her saat için tek gerçek gözlem.
    benzersiz = (
        tum.drop_duplicates("target_datetime")
        .sort_values("target_datetime")
        .set_index("target_datetime")
    )

    session_series = benzersiz[
        "session_start_count"
    ].astype(float)

    energy_series = benzersiz[
        "started_session_energy_kwh"
    ].astype(float)

    load = pd.read_csv(
        LOAD_PATH,
        parse_dates=["datetime"]
    )

    load = load.set_index("datetime").sort_index()
    load = load[load.index < TEST_START]

    return tum, session_series, energy_series, load


# ============================================================
# 3. ORTAK B0 PROFİL TAHMİNİ
# ============================================================

def b0_profile_prediction(training, evaluation, target):

    training_unique = training.drop_duplicates(
        "target_datetime"
    )

    profile = (
        training_unique
        .groupby(["target_weekday", "target_hour"])[target]
        .mean()
    )

    general_mean = training_unique[target].mean()

    if pd.isna(general_mean) or general_mean <= 0:
        raise ValueError(
            f"{target} için genel ortalama hesaplanamadı."
        )

    keys = pd.MultiIndex.from_arrays(
        [
            evaluation["target_weekday"].astype(int),
            evaluation["target_hour"].astype(int),
        ]
    )

    profile_values = profile.reindex(keys).to_numpy(dtype=float)

    level_column = f"{target}_rolling_mean_168"
    level_values = evaluation[level_column].to_numpy(dtype=float)

    prediction = (
        profile_values / general_mean
    ) * level_values

    return np.maximum(prediction, 0)


# ============================================================
# 4. OTURUM VE BAŞLANGIÇ ENERJİSİ İÇİN B0 + S1
# ============================================================

def forecast_target(tum, series, target, level_column):

    calibration_start = (
        TEST_START - pd.Timedelta(weeks=CALIBRATION_WEEKS)
    )

    previous = tum[
        tum["target_datetime"] < calibration_start
    ].copy()

    calibration = tum[
        (tum["target_datetime"] >= calibration_start)
        & (tum["target_datetime"] < TEST_START)
    ].copy()

    if previous.empty or calibration.empty:
        raise ValueError(
            f"{target} için kalibrasyon verisi yetersiz."
        )

    b_cal = b0_profile_prediction(
        previous,
        calibration,
        target
    )

    actual_cal = calibration[target].to_numpy(dtype=float)

    m = float(np.mean(actual_cal))

    scale_cal = np.sqrt(
        np.maximum(b_cal + 0.05 * m, 1e-9)
    )

    standardized_errors = (
        actual_cal - b_cal
    ) / scale_cal

    z_q = float(
        np.quantile(standardized_errors, Q_LEVEL)
    )

    last_values = series.loc[
        series.index < TEST_START
    ].tail(168)

    if len(last_values) < 168:
        raise ValueError(
            f"{target} için son 168 saatlik geçmiş veri eksik."
        )

    origin_level = float(last_values.mean())

    future_dates = pd.date_range(
        start=TEST_START,
        periods=FORECAST_HOURS,
        freq="h"
    )

    future = pd.DataFrame({
        "target_datetime": future_dates,
        "target_weekday": future_dates.dayofweek,
        "target_hour": future_dates.hour,
        level_column: origin_level,
    })

    b0 = b0_profile_prediction(
        tum,
        future,
        target
    )

    scale_future = np.sqrt(
        np.maximum(b0 + 0.05 * m, 1e-9)
    )

    s1 = np.maximum(
        0,
        b0 + z_q * scale_future
    )

    result = pd.DataFrame({
        "Zaman": future_dates,
        "Saat": future_dates.strftime("%d.%m.%Y %H:%M"),
        "B0_Tahmin": b0,
        "S1_Ust_Sinir": s1,
        "Belirsizlik_Payi": s1 - b0,
    })

    info = {
        "calibration_start": calibration_start,
        "calibration_end": TEST_START,
        "calibration_count": len(calibration),
        "calibration_mean": m,
        "z_q": z_q,
        "origin_level": origin_level,
    }

    return result, info


# ============================================================
# 5. YAKLAŞIK SAATLİK YÜK İÇİN B0
# ============================================================

def load_profile_prediction(
    evaluation,
    load_series,
    profile_end,
):

    profile_data = load_series[
        (load_series.index >= PROFILE_START)
        & (load_series.index < profile_end)
    ]

    if profile_data.empty:
        raise ValueError(
            "Yaklaşık yük profili için geçmiş veri bulunamadı."
        )

    profile_frame = pd.DataFrame({
        "value": profile_data.to_numpy(dtype=float),
        "weekday": profile_data.index.dayofweek,
        "hour": profile_data.index.hour,
    })

    profile = (
        profile_frame
        .groupby(["weekday", "hour"])["value"]
        .mean()
    )

    general_mean = float(profile_data.mean())

    if general_mean <= 0:
        raise ValueError(
            "Yaklaşık yük profilinin genel ortalaması geçersiz."
        )

    keys = pd.MultiIndex.from_arrays(
        [
            evaluation["target_weekday"].astype(int),
            evaluation["target_hour"].astype(int),
        ]
    )

    profile_values = profile.reindex(keys).to_numpy(dtype=float)

    level_values = evaluation[
        ENERGY_LEVEL_COL
    ].to_numpy(dtype=float)

    prediction = (
        profile_values / general_mean
    ) * level_values

    return np.maximum(prediction, 0)


# ============================================================
# 6. YAKLAŞIK YÜK İÇİN S1 KALİBRASYONU
# ============================================================

def forecast_load(tum, energy_series, load, scenario):

    load_column = LOAD_SCENARIOS[scenario]

    load_series = load[load_column].dropna()

    # Mevcut doğrulama metodolojisindeki iki günlük tampon korunur.
    cutoff = TEST_START - pd.Timedelta(days=2)

    calibration_start = (
        cutoff - pd.Timedelta(weeks=CALIBRATION_WEEKS)
    )

    previous = tum[
        tum["target_datetime"] < calibration_start
    ].copy()

    calibration = tum[
        (tum["target_datetime"] >= calibration_start)
        & (tum["target_datetime"] < cutoff)
    ].copy()

    if previous.empty or calibration.empty:
        raise ValueError(
            f"{scenario} için yük kalibrasyon verisi yetersiz."
        )

    b_cal = load_profile_prediction(
        calibration,
        load_series,
        calibration_start
    )

    actual_cal = calibration[
        "target_datetime"
    ].map(load_series).to_numpy(dtype=float)

    valid = (
        np.isfinite(actual_cal)
        & np.isfinite(b_cal)
    )

    actual_cal = actual_cal[valid]
    b_cal = b_cal[valid]

    if len(actual_cal) == 0:
        raise ValueError(
            f"{scenario} için eşleşen kalibrasyon yükü bulunamadı."
        )

    m = float(np.mean(actual_cal))

    scale_cal = np.sqrt(
        np.maximum(b_cal + 0.05 * m, 1e-9)
    )

    standardized_errors = (
        actual_cal - b_cal
    ) / scale_cal

    z_q = float(
        np.quantile(standardized_errors, Q_LEVEL)
    )

    # Mevcut uygulamada kullanılan son 168 saatlik enerji seviyesi.
    last_values = energy_series.loc[
        energy_series.index < TEST_START
    ].tail(168)

    if len(last_values) < 168:
        raise ValueError(
            "Yaklaşık yük tahmini için son 168 saatlik enerji verisi eksik."
        )

    origin_level = float(last_values.mean())

    future_dates = pd.date_range(
        start=TEST_START,
        periods=FORECAST_HOURS,
        freq="h"
    )

    future = pd.DataFrame({
        "target_datetime": future_dates,
        "target_weekday": future_dates.dayofweek,
        "target_hour": future_dates.hour,
        ENERGY_LEVEL_COL: origin_level,
    })

    b0 = load_profile_prediction(
        future,
        load_series,
        cutoff
    )

    scale_future = np.sqrt(
        np.maximum(b0 + 0.05 * m, 1e-9)
    )

    s1 = np.maximum(
        0,
        b0 + z_q * scale_future
    )

    result = pd.DataFrame({
        "Zaman": future_dates,
        "Saat": future_dates.strftime("%d.%m.%Y %H:%M"),
        "B0_Tahmin": b0,
        "S1_Ust_Sinir": s1,
        "Belirsizlik_Payi": s1 - b0,
    })

    info = {
        "calibration_start": calibration_start,
        "calibration_end": cutoff,
        "calibration_count": len(actual_cal),
        "calibration_mean": m,
        "z_q": z_q,
        "origin_level": origin_level,
    }

    return result, info


# ============================================================
# 7. ORTAK GRAFİK VE SONUÇ TABLOSU
# ============================================================

def show_forecast(
    result,
    unit,
    title,
    info,
    key_suffix,
    total_unit=None
):

    if total_unit is None:
        total_unit = unit

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "24 Saatlik B0 Toplamı",
            f"{result['B0_Tahmin'].sum():,.1f} {total_unit}"
        )

    with col2:
        st.metric(
            "B0 En Yüksek Saatlik Değer",
            f"{result['B0_Tahmin'].max():,.1f} {unit}"
        )

    with col3:
        st.metric(
            "S1 En Yüksek Saatlik Değer",
            f"{result['S1_Ust_Sinir'].max():,.1f} {unit}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result["Zaman"],
            y=result["S1_Ust_Sinir"],
            mode="lines",
            name="S1 Üst Sınır",
            line=dict(width=1),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result["Zaman"],
            y=result["B0_Tahmin"],
            mode="lines+markers",
            name="B0 Noktasal Tahmin",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Tarih ve Saat",
        yaxis_title=unit,
        hovermode="x unified",
        template="plotly_white",
        height=430,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    tablo = result[
        [
            "Saat",
            "B0_Tahmin",
            "S1_Ust_Sinir",
            "Belirsizlik_Payi",
        ]
    ].copy()

    tablo.columns = [
        "Saat",
        f"B0 Tahmin ({unit})",
        f"S1 Üst Sınır ({unit})",
        f"Belirsizlik Payı ({unit})",
    ]

    st.markdown("#### Saatlik Tahmin Çizelgesi")

    st.dataframe(
        tablo.round(2),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = result.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Tahminleri CSV İndir",
        data=csv_data,
        file_name=f"saatlik_tahmin_01_09_2022_{key_suffix}.csv",
        mime="text/csv",
        key=f"download_{key_suffix}",
    )

    with st.expander("Kalibrasyon ve yöntem bilgileri"):

        st.write(
            f"Kalibrasyon başlangıcı (dahil): "
            f"{info['calibration_start']:%d.%m.%Y %H:%M}"
        )

        st.write(
            f"Kalibrasyon bitiş sınırı (hariç): "
            f"{info['calibration_end']:%d.%m.%Y %H:%M}"
        )

        son_dahil_edilen_saat = (
            info["calibration_end"] - pd.Timedelta(hours=1)
        )

        st.write(
            f"Son dahil edilen hedef saat: "
            f"{son_dahil_edilen_saat:%d.%m.%Y %H:%M}"
        )

        st.write(
            f"Kalibrasyon kayıt sayısı: "
            f"{info['calibration_count']:,}"
        )

        st.write(
            f"Kalibrasyon ortalama değeri: "
            f"{info['calibration_mean']:.2f}"
        )

        st.write(
            f"q={Q_LEVEL:.2f} standartlaştırılmış artık katsayısı: "
            f"{info['z_q']:.4f}"
        )

        st.write(
            f"Origin seviyesinin son 168 saatlik ortalaması: "
            f"{info['origin_level']:.2f}"
        )

        st.caption(
            "S1, geçmiş kalibrasyon artıklarından elde edilen "
            "q=0.90 üst belirsizlik sınırıdır. Her koşulda %90 "
            "gerçekleşme garantisi veya güvenli kapasite garantisi değildir."
        )


# ============================================================
# 8. 4. SEKME: MODEL PERFORMANSI VE KARŞILAŞTIRMA
# ============================================================

def show_performance_section():

    st.subheader("Model Performansı ve Yöntem Karşılaştırması")

    st.write(
        """
        Bu bölümde, oturum başlangıç sayısı ve başlangıç enerjisi
        tahminleri için daha önce gerçekleştirilen Haziran–Ağustos
        2022 rolling-origin doğrulama sonuçları sunulmaktadır.
        """
    )

    st.info(
        "Performans değerleri önceden tamamlanmış doğrulama "
        "çalışmalarından alınmıştır. Bu bölüm uygulama açıldığında "
        "yeniden model eğitimi veya doğrulama gerçekleştirmez."
    )

    performance_data = pd.DataFrame([
        {
            "Yöntem": "B0 Profil × Seviye",
            "Oturum MAE": 2.563,
            "Enerji MAE (kWh)": 55.01,
            "Değerlendirme": "Seçilen ana yöntem",
        },
        {
            "Yöntem": "B1 Profil Varyantı",
            "Oturum MAE": 2.5719,
            "Enerji MAE (kWh)": 55.1695,
            "Değerlendirme": "B0 alternatif yapılandırması",
        },
        {
            "Yöntem": "B2 Profil Varyantı",
            "Oturum MAE": 2.5646,
            "Enerji MAE (kWh)": 56.0030,
            "Değerlendirme": "B0 alternatif yapılandırması",
        },
        {
            "Yöntem": "B3 Profil Varyantı",
            "Oturum MAE": 2.5760,
            "Enerji MAE (kWh)": 56.0680,
            "Değerlendirme": "B0 alternatif yapılandırması",
        },
        {
            "Yöntem": "R1 Profil Varyantı",
            "Oturum MAE": 2.5737,
            "Enerji MAE (kWh)": 55.8177,
            "Değerlendirme": "B0 alternatif yapılandırması",
        },
        {
            "Yöntem": "E2 Oran Hedefli HGB",
            "Oturum MAE": 2.844,
            "Enerji MAE (kWh)": 60.43,
            "Değerlendirme": "Makine öğrenmesi",
        },
        {
            "Yöntem": "E1 Takvim İndeksleri Çıkarılmış HGB",
            "Oturum MAE": 2.872,
            "Enerji MAE (kWh)": 61.10,
            "Değerlendirme": "Makine öğrenmesi",
        },
        {
            "Yöntem": "Statik Profil",
            "Oturum MAE": 3.194,
            "Enerji MAE (kWh)": 64.15,
            "Değerlendirme": "Referans yöntem",
        },
        {
            "Yöntem": "REF_45 HGB (Rolling-origin)",
            "Oturum MAE": 3.253,
            "Enerji MAE (kWh)": 71.45,
            "Değerlendirme": "İlk referans model",
        },
    ])

    st.markdown("#### Rolling-origin doğrulama sonuçları")

    st.dataframe(
        performance_data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "MAE: Ortalama Mutlak Hata. Daha düşük değer daha düşük "
        "ortalama tahmin hatasını ifade eder. B0 varyantları için "
        "tek bir değer yerine doğrulamada gözlenen aralık verilmiştir."
    )

    st.markdown("#### Oturum başlangıç sayısı – MAE karşılaştırması")

    session_methods = [
        "B0 Profil × Seviye",
        "B1 Profil Varyantı",
        "B2 Profil Varyantı",
        "B3 Profil Varyantı",
        "R1 Profil Varyantı",
        "E2 Oran Hedefli HGB",
        "E1 Takvim İndeksleri Çıkarılmış HGB",
        "Statik Profil",
        "REF_45 HGB (Rolling-origin)",
    ]

    session_mae = [
        2.5632,
        2.5719,
        2.5646,
        2.5760,
        2.5737,
        2.8440,
        2.8716,
        3.1939,
        3.2526,
    ]

    fig_session = go.Figure(
        go.Bar(
            x=session_methods,
            y=session_mae,
            text=[f"{x:.3f}" for x in session_mae],
            textposition="outside",
        )
    )

    fig_session.update_layout(
        title="Oturum Başlangıç Sayısı MAE",
        xaxis_title="Tahmin yöntemi",
        yaxis_title="MAE (oturum)",
        template="plotly_white",
        height=430,
    )

    st.plotly_chart(
        fig_session,
        use_container_width=True
    )

    st.markdown("#### Başlangıç enerjisi – MAE karşılaştırması")

    energy_methods = [
        "B0 Profil × Seviye",
        "B1 Profil Varyantı",
        "B2 Profil Varyantı",
        "B3 Profil Varyantı",
        "R1 Profil Varyantı",
        "E2 Oran Hedefli HGB",
        "E1 Takvim İndeksleri Çıkarılmış HGB",
        "Statik Profil",
        "REF_45 HGB (Rolling-origin)",
    ]

    energy_mae = [
        55.0065,
        55.1695,
        56.0030,
        56.0680,
        55.8177,
        60.4289,
        61.1026,
        64.1546,
        71.4464,
    ]

    fig_energy = go.Figure(
        go.Bar(
            x=energy_methods,
            y=energy_mae,
            text=[f"{x:.2f}" for x in energy_mae],
            textposition="outside",
        )
    )

    fig_energy.update_layout(
        title="Başlangıç Enerjisi MAE",
        xaxis_title="Tahmin yöntemi",
        yaxis_title="MAE (kWh)",
        template="plotly_white",
        height=430,
    )

    st.plotly_chart(
        fig_energy,
        use_container_width=True
    )

    st.divider()

    st.markdown("#### İlk HGB referans modelinin Eylül 2022 değerlendirmesi")

    st.write(
        """
        İlk HistGradientBoosting referans modeli ayrıca Eylül 2022
        döneminde değerlendirilmiştir. Bu değerlendirme, Haziran–Ağustos
        rolling-origin karşılaştırmasından farklı bir deney aşamasıdır.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Oturum Sayısı MAE",
            "2,9489"
        )
        st.metric(
            "Oturum Sayısı WAPE",
            "%34,03"
        )

    with col2:
        st.metric(
            "Başlangıç Enerjisi MAE",
            "61,5374 kWh"
        )
        st.metric(
            "Başlangıç Enerjisi WAPE",
            "%39,88"
        )

    st.caption(
        "Önemli: Eylül 2022, ilk HGB referans modelinin değerlendirmesinde "
        "kullanılmıştır. Bu nedenle tüm proje açısından hiç görülmemiş "
        "bağımsız test dönemi olarak nitelendirilmemelidir. Bu uygulamadaki "
        "B0 ve S1 tahmin hesapları ise Eylül gözlemlerini kullanmaz."
    )

    st.success(
        """
        **Yöntem seçimi:** Haziran–Ağustos rolling-origin değerlendirmesinde
        B0 Profil × Seviye yöntemi, hem oturum başlangıç sayısı hem de
        başlangıç enerjisi için karşılaştırılan yöntemler arasında en
        düşük ortalama MAE değerini vermiştir. Bu nedenle uygulamanın
        ana noktasal tahmin yöntemi olarak seçilmiştir.
        """
    )

    with st.expander("Performans ölçütlerinin açıklaması"):

        st.markdown(
            """
            - **MAE (Mean Absolute Error):** Gerçek değerler ile tahminler
              arasındaki mutlak farkların ortalamasıdır.
            - **WAPE (Weighted Absolute Percentage Error):** Mutlak
              hataların gerçek değerler toplamına oranlanmasıyla
              hesaplanan yüzdesel hata ölçütüdür.
            - **Rolling-origin doğrulama:** Tahmin başlangıç noktasının
              zaman içinde ilerletildiği ve her aşamada yalnızca geçmiş
              verilerin kullanıldığı değerlendirme yaklaşımıdır.

            MAE değerleri farklı hedeflerin birimleri farklı olduğu için
            birbirleriyle doğrudan büyüklük karşılaştırması amacıyla
            kullanılmamalıdır.
            """
        )


# ============================================================
# 9. 5. SEKME: TALEP YOĞUNLUĞU ANALİZİ
# ============================================================

def show_demand_profile(session_series, energy_series):

    st.subheader("Geçmiş Verilere Göre Talep Yoğunluğu")

    st.write(
        """
        Bu bölümde, Eylül 2022 öncesindeki saatlik gözlemler kullanılarak
        gün içi ve haftanın günlerine göre talep örüntüleri incelenmektedir.
        Amaç, geçmiş dönemde talebin hangi saatlerde yoğunlaştığını ve
        hangi zaman aralıklarında daha düşük kaldığını göstermektir.
        """
    )

    st.info(
        "Analiz yalnızca Eylül 2022 öncesindeki tarihsel verileri "
        "kullanır. Eylül 2022 dönemi yoğunluk profiline dahil edilmemiştir."
    )

    session_frame = pd.DataFrame({
        "deger": session_series,
        "weekday": session_series.index.dayofweek,
        "hour": session_series.index.hour,
    })

    energy_frame = pd.DataFrame({
        "deger": energy_series,
        "weekday": energy_series.index.dayofweek,
        "hour": energy_series.index.hour,
    })

    if session_frame.empty or energy_frame.empty:
        st.warning(
            "Talep yoğunluğu analizi için yeterli tarihsel veri bulunamadı."
        )
        return

    # --------------------------------------------------------
    # Saatlik ortalama profiller
    # --------------------------------------------------------

    session_hourly = (
        session_frame.groupby("hour")["deger"]
        .mean()
        .reindex(range(24))
    )

    energy_hourly = (
        energy_frame.groupby("hour")["deger"]
        .mean()
        .reindex(range(24))
    )

    session_peak_hour = int(session_hourly.idxmax())
    session_low_hour = int(session_hourly.idxmin())

    energy_peak_hour = int(energy_hourly.idxmax())
    energy_low_hour = int(energy_hourly.idxmin())

    st.markdown("#### Gün içi ortalama talep profili")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Oturum Yoğunluğu En Yüksek Saat",
            f"{session_peak_hour:02d}:00"
        )

    with c2:
        st.metric(
            "Oturum Yoğunluğu En Düşük Saat",
            f"{session_low_hour:02d}:00"
        )

    with c3:
        st.metric(
            "Başlangıç Enerjisi En Yüksek Saat",
            f"{energy_peak_hour:02d}:00"
        )

    with c4:
        st.metric(
            "Başlangıç Enerjisi En Düşük Saat",
            f"{energy_low_hour:02d}:00"
        )

    fig_session_hour = go.Figure()

    fig_session_hour.add_trace(
        go.Bar(
            x=[f"{h:02d}:00" for h in range(24)],
            y=session_hourly.values,
            name="Ortalama oturum başlangıcı",
        )
    )

    fig_session_hour.update_layout(
        title="Saatlere Göre Ortalama Oturum Başlangıç Sayısı",
        xaxis_title="Saat",
        yaxis_title="Ortalama oturum/saat",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig_session_hour,
        use_container_width=True
    )

    fig_energy_hour = go.Figure()

    fig_energy_hour.add_trace(
        go.Bar(
            x=[f"{h:02d}:00" for h in range(24)],
            y=energy_hourly.values,
            name="Başlangıç saatine atfedilen enerji",
        )
    )

    fig_energy_hour.update_layout(
        title="Saatlere Göre Ortalama Başlangıç Enerjisi",
        xaxis_title="Saat",
        yaxis_title="Ortalama enerji (kWh)",
        template="plotly_white",
        height=400,
    )

    st.plotly_chart(
        fig_energy_hour,
        use_container_width=True
    )

    st.warning(
        "Başlangıç enerjisi grafiği, oturumların toplam enerjisinin "
        "başlangıç saatine atfedilmesine dayanır. Gerçek şebeke yükünün "
        "saatlik ölçümü olarak yorumlanmamalıdır."
    )

    st.divider()

    # --------------------------------------------------------
    # Haftanın günü ve saat ısı haritaları
    # --------------------------------------------------------

    st.markdown("#### Haftanın günlerine ve saatlere göre yoğunluk")

    session_heatmap = (
        session_frame.pivot_table(
            index="weekday",
            columns="hour",
            values="deger",
            aggfunc="mean"
        )
        .reindex(index=range(7), columns=range(24))
    )

    energy_heatmap = (
        energy_frame.pivot_table(
            index="weekday",
            columns="hour",
            values="deger",
            aggfunc="mean"
        )
        .reindex(index=range(7), columns=range(24))
    )

    fig_session_heat = go.Figure(
        data=go.Heatmap(
            z=session_heatmap.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=WEEKDAY_NAMES,
            colorbar=dict(title="Oturum/saat"),
            hovertemplate=(
                "Gün: %{y}<br>"
                "Saat: %{x}<br>"
                "Ortalama oturum: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig_session_heat.update_layout(
        title="Oturum Başlangıç Sayısı – Gün ve Saat Isı Haritası",
        xaxis_title="Saat",
        yaxis_title="Haftanın günü",
        template="plotly_white",
        height=430,
    )

    st.plotly_chart(
        fig_session_heat,
        use_container_width=True
    )

    fig_energy_heat = go.Figure(
        data=go.Heatmap(
            z=energy_heatmap.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=WEEKDAY_NAMES,
            colorbar=dict(title="kWh"),
            hovertemplate=(
                "Gün: %{y}<br>"
                "Saat: %{x}<br>"
                "Ortalama enerji: %{z:.2f} kWh"
                "<extra></extra>"
            ),
        )
    )

    fig_energy_heat.update_layout(
        title="Başlangıç Enerjisi – Gün ve Saat Isı Haritası",
        xaxis_title="Saat",
        yaxis_title="Haftanın günü",
        template="plotly_white",
        height=430,
    )

    st.plotly_chart(
        fig_energy_heat,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Haftanın günlerine göre ortalamalar
    # --------------------------------------------------------

    st.markdown("#### Haftanın günlerine göre ortalama talep")

    session_weekday = (
        session_frame.groupby("weekday")["deger"]
        .mean()
        .reindex(range(7))
    )

    energy_weekday = (
        energy_frame.groupby("weekday")["deger"]
        .mean()
        .reindex(range(7))
    )

    weekday_table = pd.DataFrame({
        "Haftanın Günü": WEEKDAY_NAMES,
        "Ortalama Oturum Başlangıcı (oturum/saat)": (
            session_weekday.values
        ),
        "Ortalama Başlangıç Enerjisi (kWh)": (
            energy_weekday.values
        ),
    })

    st.dataframe(
        weekday_table.round(2),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Yoğunluk değerleri, ilgili saat veya gün için tarihsel "
        "gözlemlerin aritmetik ortalamasıdır. Gelecekte aynı talebin "
        "kesin olarak gerçekleşeceği anlamına gelmez."
    )


# ============================================================
# 10. STREAMLIT ANA EKRAN
# ============================================================

def main():

    st.title(
        "⚡ Elektrikli Araç Şarj İstasyonlarında Saatlik Talep Tahmini"
    )

    st.markdown(
        """
        Bu karar destek sistemi, geçmiş elektrikli araç şarj kayıtlarından
        yararlanarak saatlik oturum başlangıç sayısını, başlangıç saatine
        atfedilen enerji miktarını ve farklı dağıtım varsayımlarına dayalı
        yaklaşık saatlik yük profilini tahmin eder. Ayrıca tahmin
        yöntemlerinin performansını ve geçmiş talep yoğunluğunu inceleme
        olanağı sunar.
        """
    )

    st.info(
        "Tarihsel örnek tahmin: 01.09.2022 00:00–23:00. "
        "Bu ekran gerçek zamanlı tahmin veya şebeke kontrolü yapmaz."
    )

    try:
        tum, session_series, energy_series, load = load_data()

        session_result, session_info = forecast_target(
            tum,
            session_series,
            "session_start_count",
            "session_start_count_rolling_mean_168",
        )

        energy_result, energy_info = forecast_target(
            tum,
            energy_series,
            "started_session_energy_kwh",
            ENERGY_LEVEL_COL,
        )

    except Exception as error:
        st.error(
            f"Veri veya hedef tahmini yüklenemedi: {error}"
        )
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚗 Oturum Başlangıç Tahmini",
        "🔋 Başlangıç Enerjisi Tahmini",
        "⚡ Yaklaşık Saatlik Yük Tahmini",
        "📊 Model Performansı",
        "🕒 Talep Yoğunluğu Analizi",
    ])

    # --------------------------------------------------------
    # TAB 1: OTURUM SAYISI
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Saatlik Şarj Oturumu Başlangıç Sayısı"
        )

        st.write(
            "Her saat içinde başlayan şarj oturumlarının "
            "24 saatlik tahmini gösterilmektedir."
        )

        show_forecast(
            session_result,
            "oturum/saat",
            "Saatlik Oturum Başlangıç Sayısı: B0 ve S1",
            session_info,
            "oturum",
            total_unit="oturum",
        )

    # --------------------------------------------------------
    # TAB 2: BAŞLANGIÇ ENERJİSİ
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Başlangıç Saatine Atfedilen Enerji Tahmini"
        )

        st.warning(
            "Bu çıktı, her şarj oturumunun toplam enerji tüketimini "
            "oturumun başladığı saate atfeder. Enerjinin mutlaka "
            "başlangıç saatinde tüketildiğini veya şebekeye o saatte "
            "aktarıldığını göstermez."
        )

        show_forecast(
            energy_result,
            "kWh",
            "Başlangıç Saatine Atfedilen Enerji: B0 ve S1",
            energy_info,
            "baslangic_enerjisi",
            total_unit="kWh",
        )

    # --------------------------------------------------------
    # TAB 3: YAKLAŞIK SAATLİK YÜK
    # --------------------------------------------------------

    with tab3:

        st.subheader(
            "Varsayımsal Yaklaşık Saatlik Yük Tahmini"
        )

        st.write(
            """
            Şarj oturumlarında tüketilen enerjinin zamana nasıl
            dağıldığını incelemek için üç farklı varsayım
            karşılaştırılmaktadır. Senaryo seçimi, tahmin edilen
            yaklaşık saatlik yük profilini değiştirir.
            """
        )

        scenario = st.selectbox(
            "Yük dağılım senaryosu",
            options=["U100", "F50", "F25"],
            format_func=lambda x: {
                "U100": "U100 – Enerji oturum süresine eşit yayılır",
                "F50": "F50 – Enerji ilk %50 sürede verilir",
                "F25": "F25 – Enerji ilk %25 sürede verilir",
            }[x],
        )

        scenario_descriptions = {
            "U100": (
                "Temel senaryo: Enerjinin oturum süresince eşit "
                "dağıtıldığı varsayılır."
            ),
            "F50": (
                "Enerjinin oturumun ilk %50'lik süresinde "
                "tüketildiği varsayılır."
            ),
            "F25": (
                "Enerjinin oturumun ilk %25'lik süresinde "
                "tüketildiği varsayılır."
            ),
        }

        st.caption(
            scenario_descriptions[scenario]
        )

        try:
            load_result, load_info = forecast_load(
                tum,
                energy_series,
                load,
                scenario,
            )

            st.caption(
                "Saatlik birim: kWh/saat. Bir saatlik aralıkta bu "
                "değer sayısal olarak ortalama kW değerine karşılık gelir."
            )

            show_forecast(
                load_result,
                "kWh/saat",
                f"{scenario} Senaryosu – Yaklaşık Saatlik Yük Tahmini",
                load_info,
                f"yuk_{scenario}",
                total_unit="kWh",
            )

        except Exception as error:
            st.error(
                f"Yaklaşık yük tahmini üretilemedi: {error}"
            )

        st.divider()

        st.markdown("#### Senaryoların Anlamı")

        st.markdown(
            """
            - **U100:** Enerjinin oturum süresine eşit dağıtıldığı varsayımı.
            - **F50:** Enerjinin ilk %50'lik sürede tüketildiği varsayımı.
            - **F25:** Enerjinin ilk %25'lik sürede tüketildiği varsayımı.

            Bu senaryolar ölçülmüş gerçek güç profilleri değildir.
            Gerçek şarj gücünün zamana dağılımını kesin olarak göstermez;
            farklı enerji dağıtım varsayımlarının yaklaşık yük tahminine
            etkisini incelemek amacıyla kullanılır.
            """
        )

    # --------------------------------------------------------
    # TAB 4: MODEL PERFORMANSI
    # --------------------------------------------------------

    with tab4:

        show_performance_section()

    # --------------------------------------------------------
    # TAB 5: TALEP YOĞUNLUĞU
    # --------------------------------------------------------

    with tab5:

        show_demand_profile(
            session_series,
            energy_series
        )

    # --------------------------------------------------------
    # GENEL METODOLOJİ NOTLARI
    # --------------------------------------------------------

    st.divider()

    with st.expander("📘 Veri, model ve metodoloji hakkında"):

        st.markdown(
            """
            **B0 – Noktasal tahmin**

            Haftanın günü ve saate göre geçmiş ortalama profil,
            son 168 saatlik seviye ile ölçeklendirilir. B0, makine
            öğrenmesi modeli değil, istatistiksel profil-seviye
            tahmin yaklaşımıdır.

            **S1 – Belirsizlik üst sınırı**

            Son 8 haftalık kalibrasyon dönemindeki standartlaştırılmış
            tahmin artıkları kullanılarak q=0.90 üst sınırı hesaplanır.
            Karekök ölçekleme yaklaşımı uygulanır. S1 kesin bir güvenlik
            veya kapasite garantisi değildir.

            **Kalibrasyon satırları**

            Kalibrasyon veri setinde aynı hedef saat farklı tahmin
            başlangıçlarından ve tahmin ufuklarından gelebilir.
            Bu nedenle kalibrasyon satır sayısı, benzersiz saat sayısıyla
            aynı değildir. Artıklar ortak kalibrasyon havuzunda
            değerlendirilir.

            **Veri ayrımı**

            Eylül 2022 test dönemi, bu uygulamadaki B0 profilinin ve
            S1 kalibrasyon hesabının içine alınmaz. Talep yoğunluğu
            analizi de Eylül öncesindeki saatlik gözlemlerle yapılır.

            **Kapsam sınırı**

            Tahminler veri setindeki ağ düzeyindeki gözlemleri temsil eder.
            Her bir fiziksel istasyon veya trafo için ayrı tahmin değildir.
            Projede gerçek trafo/şebeke kapasitesi ölçümü bulunmamaktadır.

            **Karar destek amacı**

            Sistem, operatörün saatlik talep eğilimlerini incelemesine
            yardımcı olan tarihsel bir tahmin prototipidir. Otomatik
            kapasite kontrolü, akıllı şarj optimizasyonu veya gerçek
            zamanlı müdahale gerçekleştirmez.
            """
        )

    st.caption(
        "Elektrikli Araç Şarj İstasyonlarında Saatlik Şarj Talebinin "
        "Makine Öğrenmesiyle Tahmini ve Operasyonel Karar Destek Sistemi"
    )


if __name__ == "__main__":
    main()