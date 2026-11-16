
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ChargingRecords.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

# Tarihleri birleştirerek tam zaman damgası oluştur
df["StartFull"] = pd.to_datetime(
    df["StartDay"] + " " + df["StartTime"],
    errors="coerce"
)

df["EndFull"] = pd.to_datetime(
    df["EndDay"] + " " + df["EndTime"],
    errors="coerce"
)

# 1. Tekrarlı kayıtlar
duplicates = df[df.duplicated(keep=False)]
duplicates.to_csv(
    OUTPUT_DIR / "tekrarli_kayitlar.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n1. TEKRARLI KAYITLAR")
print("Tekrarlı satır sayısı:", len(duplicates))
print("Tekrarlı kayıt grubu sayısı:", df.duplicated().sum())

# 2. Süre anomalileri
df["ConnectionMinutes"] = (
    df["EndFull"] - df["StartFull"]
).dt.total_seconds() / 60

print("\n2. SÜRE ANALİZİ")
print("Negatif Duration:", (df["Duration"] < 0).sum())
print("Sıfır Duration:", (df["Duration"] == 0).sum())
print("Bitiş başlangıçtan önce:", (df["ConnectionMinutes"] < 0).sum())

df["DurationDifference"] = (
    df["Duration"] - df["ConnectionMinutes"]
)

print("\nDuration ile bağlantı süresi farkı:")
print(df["DurationDifference"].describe())

# 3. Sorunlu süre kayıtları
duration_issues = df[
    (df["Duration"] <= 0) |
    (df["ConnectionMinutes"] <= 0)
]

duration_issues.to_csv(
    OUTPUT_DIR / "sure_anomalileri.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. Şarj cihazı bazında özellik tutarlılığı
charger_consistency = df.groupby("ChargerID").agg(
    Company_Count=("ChargerCompany", "nunique"),
    Location_Count=("Location", "nunique"),
    Type_Count=("ChargerType", "nunique"),
    Session_Count=("ChargerID", "size")
)

inconsistent_chargers = charger_consistency[
    (charger_consistency["Company_Count"] > 1) |
    (charger_consistency["Location_Count"] > 1) |
    (charger_consistency["Type_Count"] > 1)
]

print("\n3. ŞARJ CİHAZI TUTARLILIĞI")
print("Toplam ChargerID:", df["ChargerID"].nunique())
print("Tutarsız cihaz sayısı:", len(inconsistent_chargers))

inconsistent_chargers.to_csv(
    OUTPUT_DIR / "cihaz_tutarlilik_raporu.csv",
    encoding="utf-8-sig"
)

# 5. Ortalama güç analizi
valid_duration = df["Duration"] > 0

df["AveragePower_kW"] = np.nan

df.loc[valid_duration, "AveragePower_kW"] = (
    df.loc[valid_duration, "Demand"] /
    (df.loc[valid_duration, "Duration"] / 60)
)

print("\n4. ORTALAMA GÜÇ ANALİZİ")

for charger_type in sorted(df["ChargerType"].unique()):
    subset = df[
        (df["ChargerType"] == charger_type) &
        (df["AveragePower_kW"].notna())
    ]

    print(f"\nChargerType = {charger_type}")
    print("Medyan güç:", subset["AveragePower_kW"].median())
    print("Ortalama güç:", subset["AveragePower_kW"].mean())
    print("50 kW üzeri kayıt:", (subset["AveragePower_kW"] > 50).sum())
    print("350 kW üzeri kayıt:", (subset["AveragePower_kW"] > 350).sum())

# 6. Anonim kullanıcı oranı
print("\n5. KULLANICI ANALİZİ")
print("UserID = 0 kayıt sayısı:", (df["UserID"] == 0).sum())
print(
    "UserID = 0 yüzdesi:",
    round((df["UserID"] == 0).mean() * 100, 2)
)

# 7. Sonuç
print("\n" + "=" * 55)
print("DETAYLI VERİ KALİTE ANALİZİ TAMAMLANDI")
print("Oluşturulan raporlar:", OUTPUT_DIR)
print("=" * 55)
