
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "ChargingRecords.csv"

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

original_columns = df.columns.tolist()

# Sayısal alanlar
for col in ["Duration", "Demand"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Tarih ve saatleri birleştir
df["StartFull"] = pd.to_datetime(
    df["StartDay"].astype("string") + " " +
    df["StartTime"].astype("string"),
    errors="coerce"
)

df["EndFull"] = pd.to_datetime(
    df["EndDay"].astype("string") + " " +
    df["EndTime"].astype("string"),
    errors="coerce"
)

# Aynı kayıtların fazladan kopyalarını çıkar
df = df.drop_duplicates(
    subset=original_columns,
    keep="first"
).copy()

df["ConnectionMinutes"] = (
    df["EndFull"] - df["StartFull"]
).dt.total_seconds() / 60

df["DurationDifference"] = (
    df["Duration"] - df["ConnectionMinutes"]
)

# Oturum tahmini filtresi
session_mask = (
    df["StartFull"].notna()
    & df["Demand"].notna()
    & (df["Demand"] >= 0)
)

# Güç analizi filtresi
power_mask = (
    df["StartFull"].notna()
    & df["EndFull"].notna()
    & (df["Duration"] > 0)
    & (df["Demand"] >= 0)
    & (df["ConnectionMinutes"] > 0)
    & (df["DurationDifference"] <= 5)
)

# Oturum verisinde olup güç verisinde olmayan kayıtlar
excluded = df.loc[session_mask & ~power_mask].copy()

print("=" * 60)
print("GÜÇ ANALİZİNDEN DIŞARIDA KALAN KAYITLAR")
print("=" * 60)

print("Toplam oturum verisi:", int(session_mask.sum()))
print("Toplam güç analizi verisi:", int(power_mask.sum()))
print("Dışarıda kalan kayıt:", len(excluded))

total_energy = df.loc[session_mask, "Demand"].sum()
excluded_energy = excluded["Demand"].sum()

print("\n--- ENERJİ ETKİSİ ---")
print("Toplam oturum enerjisi (kWh):", round(total_energy, 2))
print("Dışarıda kalan kayıtların enerjisi (kWh):",
      round(excluded_energy, 2))

if total_energy > 0:
    print("Enerji içindeki payı (%):",
          round(excluded_energy / total_energy * 100, 4))

# Dışarıda kalma nedenlerini ayrı ayrı say
reasons = {
    "Negatif Duration": excluded["Duration"] < 0,
    "Sıfır Duration": excluded["Duration"] == 0,
    "Duration farkı > 5 dakika":
        excluded["DurationDifference"] > 5,
    "Negatif/sıfır bağlantı süresi":
        excluded["ConnectionMinutes"] <= 0,
    "Geçersiz başlangıç zamanı":
        excluded["StartFull"].isna(),
    "Geçersiz bitiş zamanı":
        excluded["EndFull"].isna(),
    "Eksik veya negatif Demand":
        excluded["Demand"].isna() | (excluded["Demand"] < 0)
}

print("\n--- DIŞARIDA KALMA NEDENLERİ ---")

for name, mask in reasons.items():
    count = int(mask.sum())
    energy = excluded.loc[mask, "Demand"].sum()

    print(
        f"{name}: {count} kayıt | "
        f"Enerji: {energy:.2f} kWh"
    )

# Aynı kaydın birden fazla koşula takılıp takılmadığını göster
reason_columns = []

for name, mask in reasons.items():
    col = "flag_" + str(len(reason_columns))
    excluded[col] = mask
    reason_columns.append(col)

print("\n--- BİRDEN FAZLA KOŞULA TAKILAN KAYITLAR ---")

excluded["reason_count"] = (
    excluded[reason_columns].sum(axis=1)
)

print(
    excluded["reason_count"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n--- İLK 15 DIŞARIDA KALAN KAYIT ---")

show_columns = [
    "Duration",
    "Demand",
    "ConnectionMinutes",
    "DurationDifference",
    "StartFull",
    "EndFull"
]

print(
    excluded[show_columns]
    .head(15)
    .to_string(index=False)
)

print("\nKontrol tamamlandı.")
