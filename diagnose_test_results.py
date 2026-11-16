
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs_modeling"
FEATURE_DIR = BASE_DIR / "outputs_feature_engineering"

pred = pd.read_csv(
    OUTPUT_DIR / "test_tahminleri.csv",
    parse_dates=["origin_datetime", "target_datetime"]
)

test = pd.read_csv(
    FEATURE_DIR / "test_verisi.csv",
    parse_dates=["origin_datetime", "target_datetime"]
)

print("=" * 65)
print("TEST SONUÇLARI ANORMALLİK İNCELEMESİ")
print("=" * 65)

# --------------------------------------------------
# 1. GERÇEK VE TAHMİN DEĞERLERİNİN DAĞILIMI
# --------------------------------------------------

for target in pred["target"].unique():

    part = pred[pred["target"] == target]

    print("\nHEDEF:", target)

    print("\nGerçek değerler:")
    print(part["actual"].describe())

    print("\nTahmin değerleri:")
    print(part["predicted"].describe())

    print("\nEn büyük 10 tahmin:")
    print(
        part.nlargest(10, "predicted")[
            [
                "origin_datetime",
                "target_datetime",
                "forecast_horizon",
                "actual",
                "predicted"
            ]
        ].to_string(index=False)
    )

# --------------------------------------------------
# 2. OTURUM SAYISI İÇİN AYRINTILI KONTROL
# --------------------------------------------------

count_data = pred[
    pred["target"] == "session_start_count"
].copy()

print("\n" + "=" * 65)
print("OTURUM SAYISI ÖZEL KONTROLÜ")
print("=" * 65)

print("Gerçek minimum:", count_data["actual"].min())
print("Gerçek maksimum:", count_data["actual"].max())

print("Tahmin minimum:", count_data["predicted"].min())
print("Tahmin maksimum:", count_data["predicted"].max())

print(
    "Gerçek değer 38 üzerinde:",
    (count_data["actual"] > 38).sum()
)

print(
    "Tahmin değeri 38 üzerinde:",
    (count_data["predicted"] > 38).sum()
)

# --------------------------------------------------
# 3. TAHMİN UFUKLARINA GÖRE HATA
# --------------------------------------------------

count_data["absolute_error"] = (
    count_data["actual"] - count_data["predicted"]
).abs()

horizon_summary = (
    count_data
    .groupby("forecast_horizon")
    .agg(
        sample_count=("actual", "size"),
        actual_mean=("actual", "mean"),
        actual_max=("actual", "max"),
        predicted_mean=("predicted", "mean"),
        predicted_max=("predicted", "max"),
        MAE=("absolute_error", "mean")
    )
)

print("\nTahmin ufkuna göre oturum sayısı:")
print(horizon_summary.round(3).to_string())

# --------------------------------------------------
# 4. TARİHE GÖRE HATA
# --------------------------------------------------

count_data["target_day"] = (
    count_data["target_datetime"].dt.date
)

daily_summary = (
    count_data
    .groupby("target_day")
    .agg(
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
        MAE=("absolute_error", "mean")
    )
)

print("\nEn yüksek hatalı 10 gün:")
print(
    daily_summary
    .sort_values("MAE", ascending=False)
    .head(10)
    .round(3)
    .to_string()
)

# --------------------------------------------------
# 5. TEST VERİSİ İLE TAHMİN DOSYASI KONTROLÜ
# --------------------------------------------------

print("\n" + "=" * 65)
print("VERİ EŞLEŞMESİ KONTROLÜ")
print("=" * 65)

print("Test dosyası satırı:", len(test))
print("Tahmin dosyası satırı:", len(pred))

print(
    "Test oturum sayısı min:",
    test["session_start_count"].min()
)

print(
    "Test oturum sayısı max:",
    test["session_start_count"].max()
)

print(
    "Test enerji min:",
    test["started_session_energy_kwh"].min()
)

print(
    "Test enerji max:",
    test["started_session_energy_kwh"].max()
)

print("\nTest dosyası sütunları:")
print(test.columns.tolist())

print("\nTahmin dosyası hedef dağılımı:")
print(pred["target"].value_counts())

print("\nİnceleme tamamlandı.")
