
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "outputs_modeling" / "test_tahminleri.csv"

df = pd.read_csv(FILE)

energy = df[
    (df["target"] == "started_session_energy_kwh") &
    (df["forecast_horizon"] == 24)
].copy()

energy["target_datetime"] = pd.to_datetime(
    energy["target_datetime"]
)

energy["error"] = energy["actual"] - energy["predicted"]
energy["absolute_error"] = energy["error"].abs()

print("=" * 60)
print("ENERJİ TAHMİNİ - 24 SAATLİK UFUK ANALİZİ")
print("=" * 60)

print("\nKayıt sayısı:", len(energy))
print(
    "Tarih aralığı:",
    energy["target_datetime"].min(),
    "-",
    energy["target_datetime"].max()
)

actual = energy["actual"]
pred = energy["predicted"]

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred) ** 2))
wape = (
    np.sum(np.abs(actual - pred)) /
    np.sum(np.abs(actual)) * 100
)
smape = np.mean(
    2 * np.abs(actual - pred) /
    (np.abs(actual) + np.abs(pred) + 1e-8)
) * 100

print("\n1. GENEL PERFORMANS")
print(f"MAE: {mae:.2f} kWh")
print(f"RMSE: {rmse:.2f} kWh")
print(f"WAPE: %{wape:.2f}")
print(f"sMAPE: %{smape:.2f}")

print("\n2. TAHMİN DAĞILIMI")
print(f"Gerçek ortalama: {actual.mean():.2f}")
print(f"Tahmin ortalama: {pred.mean():.2f}")
print(f"Gerçek maksimum: {actual.max():.2f}")
print(f"Tahmin maksimum: {pred.max():.2f}")

print("\n3. EN YÜKSEK GERÇEK ENERJİ SAATLERİ")
print(
    energy.nlargest(10, "actual")[
        ["target_datetime", "actual", "predicted", "error"]
    ].to_string(index=False)
)

print("\n4. ZİRVE ANALİZİ")

for threshold in [200, 300, 400, 500]:
    subset = energy[energy["actual"] >= threshold]

    if len(subset):
        print(
            f"\nGerçek >= {threshold} kWh"
            f"\nKayıt: {len(subset)}"
            f"\nOrtalama gerçek: {subset['actual'].mean():.2f}"
            f"\nOrtalama tahmin: {subset['predicted'].mean():.2f}"
            f"\nOrtalama hata (gerçek-tahmin): "
            f"{subset['error'].mean():.2f}"
        )

print("\n5. SAAT BAZINDA HATA")

energy["hour"] = energy["target_datetime"].dt.hour

hourly = energy.groupby("hour").agg(
    actual_mean=("actual", "mean"),
    predicted_mean=("predicted", "mean"),
    mean_error=("error", "mean"),
    MAE=("absolute_error", "mean")
)

print(hourly.round(2).to_string())

OUTPUT = BASE_DIR / "outputs_modeling" / "enerji_24_saat_ufuk_analizi.csv"
hourly.to_csv(OUTPUT, encoding="utf-8-sig")

print("\nSaatlik analiz kaydedildi:", OUTPUT)
print("\nAnaliz tamamlandı.")
