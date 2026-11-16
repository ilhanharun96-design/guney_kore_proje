
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJE DOSYA YOLLARI
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ChargingRecords.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("ELEKTRİKLİ ARAÇ ŞARJ TALEBİ - VERİ İNCELEME")
print("=" * 60)

# ==========================================
# 2. VERİ SETİNİ OKUMA
# ==========================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"CSV dosyası bulunamadı: {DATA_PATH}"
    )

try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="cp949", low_memory=False)

print("\n[1] VERİ SETİ GENEL BİLGİLERİ")

print("Satır sayısı:", df.shape[0])
print("Sütun sayısı:", df.shape[1])

print("\nSütun isimleri:")
print(df.columns.tolist())

# ==========================================
# 3. VERİ TİPLERİ
# ==========================================

print("\n[2] SÜTUNLARIN VERİ TİPLERİ")
print(df.dtypes)

# ==========================================
# 4. EKSİK DEĞER KONTROLÜ
# ==========================================

print("\n[3] EKSİK DEĞER ANALİZİ")

missing = df.isnull().sum()

missing_report = pd.DataFrame({
    "Eksik_Deger_Sayisi": missing,
    "Eksik_Deger_Yuzdesi": (missing / len(df)) * 100
})

print(missing_report)

missing_report.to_csv(
    OUTPUT_DIR / "eksik_deger_raporu.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 5. TEKRAR EDEN KAYITLAR
# ==========================================

print("\n[4] TEKRAR EDEN KAYITLAR")

duplicate_count = df.duplicated().sum()

print("Tamamen aynı kayıt sayısı:", duplicate_count)

# ==========================================
# 6. SAYISAL DEĞİŞKENLERİN İSTATİSTİKLERİ
# ==========================================

print("\n[5] SAYISAL DEĞİŞKENLERİN İSTATİSTİKLERİ")

print(df.describe().T)

df.describe().T.to_csv(
    OUTPUT_DIR / "sayisal_istatistikler.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 7. SÜRE VE ENERJİ KONTROLÜ
# ==========================================

print("\n[6] SÜRE VE ENERJİ KONTROLÜ")

if "Duration" in df.columns:
    print("Negatif süreli kayıt:", (df["Duration"] < 0).sum())
    print("Sıfır süreli kayıt:", (df["Duration"] == 0).sum())

if "Demand" in df.columns:
    print("Sıfır veya negatif enerji:", (df["Demand"] <= 0).sum())
    print("80 kWh üzeri kayıt:", (df["Demand"] > 80).sum())

# ==========================================
# 8. TARİH KONTROLÜ
# ==========================================

print("\n[7] TARİH KONTROLÜ")

if "StartDay" in df.columns:
    start_dates = pd.to_datetime(
        df["StartDay"], errors="coerce"
    )

    print("İlk kayıt tarihi:", start_dates.min())
    print("Son kayıt tarihi:", start_dates.max())
    print("Geçersiz başlangıç tarihi:", start_dates.isna().sum())

if "EndDay" in df.columns:
    end_dates = pd.to_datetime(
        df["EndDay"], errors="coerce"
    )

    print("İlk bitiş tarihi:", end_dates.min())
    print("Son bitiş tarihi:", end_dates.max())
    print("Geçersiz bitiş tarihi:", end_dates.isna().sum())

# ==========================================
# 9. KATEGORİK DEĞİŞKENLER
# ==========================================

print("\n[8] KATEGORİK DEĞİŞKENLER")

for column in ["Location", "ChargerType", "ChargerCompany"]:
    if column in df.columns:
        print(f"\n{column} dağılımı:")
        print(df[column].value_counts(dropna=False))

# ==========================================
# 10. SONUÇ
# ==========================================

print("\n" + "=" * 60)
print("İLK VERİ İNCELEMESİ TAMAMLANDI")
print("Raporlar outputs klasörüne kaydedildi.")
print("=" * 60)
