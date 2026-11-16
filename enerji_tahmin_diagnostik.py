
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent

FILE = (
    BASE_DIR
    / "outputs_modeling"
    / "test_tahminleri.csv"
)

df = pd.read_csv(FILE)

print("=" * 60)
print("ENERJİ TAHMİNİ DETAYLI İNCELEME")
print("=" * 60)

energy = df[
    df["target"] == "started_session_energy_kwh"
].copy()

energy["target_datetime"] = pd.to_datetime(
    energy["target_datetime"]
)

actual = energy["actual"]
pred = energy["predicted"]

print("\n1. TAHMİN DAĞILIMI")

print("Gerçek maksimum:", actual.max())
print("Tahmin maksimum:", pred.max())
print("Gerçek ortalama:", actual.mean())
print("Tahmin ortalama:", pred.mean())

print("\n2. GERÇEK ZİRVELER")

print(
    energy.nlargest(10, "actual")[
        ["target_datetime", "actual", "predicted"]
    ].to_string(index=False)
)

print("\n3. EN BÜYÜK TAHMİN HATALARI")

energy["absolute_error"] = (
    energy["actual"] - energy["predicted"]
).abs()

energy["underprediction"] = (
    energy["actual"] - energy["predicted"]
)

print(
    energy.nlargest(10, "absolute_error")[
        [
            "target_datetime",
            "actual",
            "predicted",
            "underprediction"
        ]
    ].to_string(index=False)
)

print("\n4. ZİRVE YAKALAMA ANALİZİ")

for threshold in [200, 300, 400, 500]:
    subset = energy[energy["actual"] >= threshold]

    if len(subset) == 0:
        continue

    print(
        f"\nGerçek >= {threshold} kWh:"
        f"\nKayıt sayısı: {len(subset)}"
        f"\nOrtalama gerçek: {subset['actual'].mean():.2f}"
        f"\nOrtalama tahmin: {subset['predicted'].mean():.2f}"
        f"\nOrtalama eksik tahmin: "
        f"{(subset['actual'] - subset['predicted']).mean():.2f}"
    )

print("\n5. SAAT BAZINDA ORTALAMA SAPMA")

energy["hour"] = energy["target_datetime"].dt.hour

hourly = energy.groupby("hour").agg(
    actual_mean=("actual", "mean"),
    predicted_mean=("predicted", "mean"),
    mean_error=("underprediction", "mean"),
    max_actual=("actual", "max"),
    max_predicted=("predicted", "max")
)

print(hourly.round(2).to_string())

print("\nİnceleme tamamlandı.")
