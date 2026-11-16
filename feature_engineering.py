
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. DOSYA YOLLARI
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "outputs_cleaning"
    / "saatlik_oturum_tahmin_verisi.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs_feature_engineering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. VERİYİ OKU VE KONTROL ET
# --------------------------------------------------

print("=" * 65)
print("FEATURE ENGINEERING - 24 SAATLİK TALEP TAHMİNİ")
print("=" * 65)

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "Hour",
    "session_start_count",
    "started_session_energy_kwh"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Eksik sütunlar bulundu: {missing_columns}"
    )

df["DateTime"] = pd.to_datetime(
    df["Hour"],
    errors="coerce"
)

df = df.dropna(subset=["DateTime"]).copy()

df["session_start_count"] = pd.to_numeric(
    df["session_start_count"],
    errors="coerce"
)

df["started_session_energy_kwh"] = pd.to_numeric(
    df["started_session_energy_kwh"],
    errors="coerce"
)

df = df.sort_values("DateTime").reset_index(drop=True)

if df[
    ["session_start_count", "started_session_energy_kwh"]
].isna().any().any():
    raise ValueError("Hedef değişkenlerde eksik değer var.")

# Saatlik zaman sürekliliğini kontrol et
time_difference = df["DateTime"].diff().dropna()

if not (time_difference == pd.Timedelta(hours=1)).all():
    raise ValueError(
        "Veride kesintisiz saatlik zaman dizisi bulunmuyor. "
        "Eksik veya tekrarlı saatleri inceleyin."
    )

print(f"Veri satırı: {len(df)}")
print(f"Başlangıç: {df['DateTime'].min()}")
print(f"Bitiş: {df['DateTime'].max()}")

# --------------------------------------------------
# 3. ZAMAN ÖZELLİKLERİ
# --------------------------------------------------

def add_time_features(data, prefix=""):
    """Tarih ve saat bilgilerinden takvim özellikleri üretir."""

    result = pd.DataFrame(index=data.index)

    hour = data["DateTime"].dt.hour
    weekday = data["DateTime"].dt.dayofweek
    month = data["DateTime"].dt.month
    day_of_year = data["DateTime"].dt.dayofyear

    result[f"{prefix}hour"] = hour
    result[f"{prefix}weekday"] = weekday
    result[f"{prefix}month"] = month
    result[f"{prefix}day_of_year"] = day_of_year

    result[f"{prefix}is_weekend"] = (
        weekday >= 5
    ).astype(int)

    # Döngüsel zaman özellikleri
    result[f"{prefix}hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )
    result[f"{prefix}hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    result[f"{prefix}weekday_sin"] = np.sin(
        2 * np.pi * weekday / 7
    )
    result[f"{prefix}weekday_cos"] = np.cos(
        2 * np.pi * weekday / 7
    )

    result[f"{prefix}day_of_year_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )
    result[f"{prefix}day_of_year_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return result


# --------------------------------------------------
# 4. TAHMİN BAŞLANGICI İÇİN GEÇMİŞ ÖZELLİKLER
# --------------------------------------------------

target_columns = [
    "session_start_count",
    "started_session_energy_kwh"
]

base_features = pd.DataFrame(index=df.index)

base_features["origin_datetime"] = df["DateTime"]

# Tahmin başlangıcına ait zaman özellikleri
origin_time_features = add_time_features(
    df,
    prefix="origin_"
)

base_features = pd.concat(
    [base_features, origin_time_features],
    axis=1
)

# Gecikmeli değerler:
# lag_0 = tahmin başlangıcında bilinen son değer
# lag_1 = bir saat önceki değer vb.

lags = [0, 1, 2, 3, 24, 48, 168]

for target in target_columns:

    for lag in lags:

        column_name = f"{target}_lag_{lag}"

        base_features[column_name] = (
            df[target].shift(lag)
        )

    # Geçmiş 24 ve 168 saatin hareketli istatistikleri
    for window in [24, 168]:

        base_features[
            f"{target}_rolling_mean_{window}"
        ] = (
            df[target]
            .rolling(window=window)
            .mean()
        )

        base_features[
            f"{target}_rolling_std_{window}"
        ] = (
            df[target]
            .rolling(window=window)
            .std()
        )

# --------------------------------------------------
# 5. 24 SAATLİK DOĞRUDAN TAHMİN VERİ SETİ
# --------------------------------------------------

# Her tahmin başlangıcı için sonraki 24 saat oluşturulur.
# Örneğin:
# Başlangıç: 10:00
# Hedefler: 11:00, 12:00, ... ertesi gün 10:00

HORIZON = 24

# 168 saatlik hareketli istatistik için yeterli geçmiş
# ve sonraki 24 saat için yeterli hedef bulunmalı.
first_origin = 167
last_origin = len(df) - HORIZON - 1

if last_origin < first_origin:
    raise ValueError(
        "24 saatlik tahmin veri seti oluşturmak için "
        "yeterli veri bulunmuyor."
    )

origin_indices = np.arange(
    first_origin,
    last_origin + 1
)

horizons = np.tile(
    np.arange(1, HORIZON + 1),
    len(origin_indices)
)

repeated_origins = np.repeat(
    origin_indices,
    HORIZON
)

target_indices = repeated_origins + horizons

# Tahmin başlangıcında bilinen özellikler
dataset = (
    base_features
    .iloc[repeated_origins]
    .reset_index(drop=True)
)

# Tahmin edilecek gelecek saatin özellikleri
future_dates = df.iloc[target_indices][
    ["DateTime"]
].reset_index(drop=True)

future_features = add_time_features(
    future_dates,
    prefix="target_"
)

dataset = pd.concat(
    [dataset, future_features],
    axis=1
)

# Tahmin ufku
dataset["forecast_horizon"] = horizons

# Gerçek hedef değerleri
dataset["target_datetime"] = (
    df.iloc[target_indices]["DateTime"]
    .reset_index(drop=True)
)

for target in target_columns:

    dataset[target] = (
        df.iloc[target_indices][target]
        .reset_index(drop=True)
    )

# Eksik özellikleri temizle
dataset = dataset.dropna().reset_index(drop=True)

# --------------------------------------------------
# 6. KRONOLOJİK EĞİTİM / DOĞRULAMA / TEST AYRIMI
# --------------------------------------------------

# Ayrım tahmin başlangıcına değil,
# tahmin edilen hedef tarihe göre yapılır.

target_date = pd.to_datetime(
    dataset["target_datetime"]
)

train_mask = (
    (target_date >= "2021-10-01") &
    (target_date < "2022-08-01")
)

validation_mask = (
    (target_date >= "2022-08-01") &
    (target_date < "2022-09-01")
)

test_mask = (
    (target_date >= "2022-09-01") &
    (target_date < "2022-10-01")
)

train_data = dataset.loc[train_mask].copy()
validation_data = dataset.loc[validation_mask].copy()
test_data = dataset.loc[test_mask].copy()

# --------------------------------------------------
# 7. SONUÇLARI KAYDET
# --------------------------------------------------

dataset.to_csv(
    OUTPUT_DIR / "tum_24_saatlik_tahmin_verisi.csv",
    index=False,
    encoding="utf-8-sig"
)

train_data.to_csv(
    OUTPUT_DIR / "train_verisi.csv",
    index=False,
    encoding="utf-8-sig"
)

validation_data.to_csv(
    OUTPUT_DIR / "validation_verisi.csv",
    index=False,
    encoding="utf-8-sig"
)

test_data.to_csv(
    OUTPUT_DIR / "test_verisi.csv",
    index=False,
    encoding="utf-8-sig"
)

# Özellik isimlerini ayrıca kaydet
feature_columns = [
    col for col in dataset.columns
    if col not in [
        "origin_datetime",
        "target_datetime",
        "session_start_count",
        "started_session_energy_kwh"
    ]
]

pd.DataFrame({
    "feature_name": feature_columns
}).to_csv(
    OUTPUT_DIR / "ozellik_listesi.csv",
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 8. ÖZET RAPOR
# --------------------------------------------------

summary = f"""
FEATURE ENGINEERING SONUÇ RAPORU
================================

Orijinal saatlik gözlem sayısı: {len(df)}

Tahmin ufku: {HORIZON} saat

Oluşturulan toplam örnek: {len(dataset)}

Eğitim verisi: {len(train_data)}
Doğrulama verisi: {len(validation_data)}
Test verisi: {len(test_data)}

Eğitim hedef tarih aralığı:
{train_data['target_datetime'].min()}
-
{train_data['target_datetime'].max()}

Doğrulama hedef tarih aralığı:
{validation_data['target_datetime'].min()}
-
{validation_data['target_datetime'].max()}

Test hedef tarih aralığı:
{test_data['target_datetime'].min()}
-
{test_data['target_datetime'].max()}

Özellik sayısı: {len(feature_columns)}

Tahmin hedefleri:
- session_start_count
- started_session_energy_kwh

Tahmin yaklaşımı:
Direct Multi-Horizon Forecasting

Veri ayrımı:
Kronolojik; hedef tarihine göre.
"""

with open(
    OUTPUT_DIR / "feature_engineering_ozet.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("Oluşturulan dosyalar:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)

print("\nİşlem başarıyla tamamlandı.")
