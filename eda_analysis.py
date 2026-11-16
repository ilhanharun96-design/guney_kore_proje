
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. DOSYA YOLLARI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "outputs_cleaning"
    / "saatlik_oturum_tahmin_verisi.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs_eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ELEKTRİKLİ ARAÇ ŞARJ TALEBİ - EDA ANALİZİ")
print("=" * 60)

# ============================================================
# 2. VERİYİ OKU
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Veri dosyası bulunamadı: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

print("\n[1] VERİ SETİ BİLGİLERİ")
print("-" * 60)

print("Satır sayısı:", len(df))
print("Sütunlar:", df.columns.tolist())

# Tarih sütununu belirle
date_candidates = [
    "StartFull",
    "StartHour",
    "Hour",
    "Datetime",
    "DateTime",
    "Timestamp",
    "Unnamed: 0"
]

date_col = next(
    (col for col in date_candidates if col in df.columns),
    None
)

if date_col is None:
    date_col = df.columns[0]

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df = df.dropna(subset=[date_col]).copy()

df = df.rename(columns={date_col: "DateTime"})

# Gerekli hedef sütunlar
required_cols = [
    "session_start_count",
    "started_session_energy_kwh"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(
            f"Gerekli sütun bulunamadı: {col}"
        )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required_cols).copy()
df = df.sort_values("DateTime").reset_index(drop=True)

# ============================================================
# 3. ZAMANSAL DEĞİŞKENLER
# ============================================================

df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek
df["Date"] = df["DateTime"].dt.date
df["Month"] = df["DateTime"].dt.month
df["YearMonth"] = df["DateTime"].dt.to_period("M").astype(str)

weekday_names = {
    0: "Pazartesi",
    1: "Salı",
    2: "Çarşamba",
    3: "Perşembe",
    4: "Cuma",
    5: "Cumartesi",
    6: "Pazar"
}

df["Weekday"] = df["DayOfWeek"].map(weekday_names)

# ============================================================
# 4. GENEL İSTATİSTİKLER
# ============================================================

print("\n[2] GENEL TALEP İSTATİSTİKLERİ")
print("-" * 60)

for col in required_cols:
    print(f"\n{col}")
    print(df[col].describe().round(2))

zero_hours = (df["session_start_count"] == 0).sum()

print("\nSıfır oturum bulunan saat:", zero_hours)
print(
    "Sıfır oturum oranı (%):",
    round(zero_hours / len(df) * 100, 2)
)

print(
    "Toplam oturum:",
    int(df["session_start_count"].sum())
)

print(
    "Toplam oturum enerjisi (kWh):",
    round(df["started_session_energy_kwh"].sum(), 2)
)

# ============================================================
# 5. SAATLİK TALEP GRAFİĞİ
# ============================================================

plt.figure(figsize=(15, 5))

plt.plot(
    df["DateTime"],
    df["session_start_count"],
    linewidth=0.7
)

plt.title("Saatlik Şarj Oturumu Başlangıç Sayısı")
plt.xlabel("Tarih")
plt.ylabel("Oturum Sayısı")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_saatlik_talep.png",
    dpi=300
)
plt.close()

# ============================================================
# 6. GÜN İÇİ TALEP DAĞILIMI
# ============================================================

hourly_demand = (
    df.groupby("Hour")["session_start_count"]
    .agg(["mean", "median", "std", "max"])
    .round(2)
)

print("\n[3] SAATLERE GÖRE ORTALAMA TALEP")
print("-" * 60)
print(hourly_demand)

hourly_demand.to_csv(
    OUTPUT_DIR / "saatlik_talep_istatistikleri.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 5))

plt.bar(
    hourly_demand.index,
    hourly_demand["mean"]
)

plt.title("Günün Saatlerine Göre Ortalama Şarj Talebi")
plt.xlabel("Saat")
plt.ylabel("Ortalama Oturum Sayısı")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_gun_ici_talep.png",
    dpi=300
)
plt.close()

# ============================================================
# 7. HAFTANIN GÜNLERİNE GÖRE TALEP
# ============================================================

weekday_demand = (
    df.groupby("DayOfWeek")["session_start_count"]
    .agg(["mean", "median", "max"])
    .round(2)
)

weekday_demand.index = [
    weekday_names[i] for i in weekday_demand.index
]

print("\n[4] HAFTANIN GÜNLERİNE GÖRE TALEP")
print("-" * 60)
print(weekday_demand)

weekday_demand.to_csv(
    OUTPUT_DIR / "haftalik_talep_istatistikleri.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(11, 5))

plt.bar(
    weekday_demand.index,
    weekday_demand["mean"]
)

plt.title("Haftanın Günlerine Göre Ortalama Şarj Talebi")
plt.xlabel("Gün")
plt.ylabel("Ortalama Oturum Sayısı")
plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_haftalik_talep.png",
    dpi=300
)
plt.close()

# ============================================================
# 8. AYLIK TALEP DEĞİŞİMİ
# ============================================================

monthly_demand = (
    df.groupby("YearMonth")
    .agg(
        average_sessions=("session_start_count", "mean"),
        total_sessions=("session_start_count", "sum"),
        average_energy=("started_session_energy_kwh", "mean"),
        total_energy=("started_session_energy_kwh", "sum"),
        observed_hours=("session_start_count", "count")
    )
    .round(2)
)

print("\n[5] AYLIK TALEP DEĞİŞİMİ")
print("-" * 60)
print(monthly_demand)

monthly_demand.to_csv(
    OUTPUT_DIR / "aylik_talep_istatistikleri.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(13, 5))

plt.plot(
    monthly_demand.index,
    monthly_demand["average_sessions"],
    marker="o"
)

plt.title("Aylara Göre Ortalama Saatlik Şarj Talebi")
plt.xlabel("Ay")
plt.ylabel("Saatlik Ortalama Oturum Sayısı")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_aylik_talep.png",
    dpi=300
)
plt.close()

# ============================================================
# 9. SAAT VE HAFTA GÜNÜ ISI HARİTASI
# Tam 24 saatlik günler kullanılır.
# İlk ve son kısmi günler dışarıda bırakılır.
# ============================================================

daily_hours = df.groupby("Date")["DateTime"].count()

complete_dates = daily_hours[
    daily_hours == 24
].index

complete_df = df[
    df["Date"].isin(complete_dates)
].copy()

heatmap_data = (
    complete_df.pivot_table(
        index="DayOfWeek",
        columns="Hour",
        values="session_start_count",
        aggfunc="mean"
    )
    .reindex(index=range(7), columns=range(24))
)

heatmap_data.index = [
    weekday_names[i] for i in heatmap_data.index
]

print("\n[6] SAAT-GÜN ISI HARİTASI VERİSİ")
print("-" * 60)
print(heatmap_data.round(2))

heatmap_data.to_csv(
    OUTPUT_DIR / "gun_saat_isi_haritasi.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(16, 6))

plt.imshow(
    heatmap_data,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(label="Ortalama Oturum Sayısı")

plt.title("Haftanın Günleri ve Saatlere Göre Şarj Yoğunluğu")
plt.xlabel("Saat")
plt.ylabel("Haftanın Günü")

plt.xticks(range(24), range(24))
plt.yticks(range(7), heatmap_data.index)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "05_gun_saat_isi_haritasi.png",
    dpi=300
)
plt.close()

# ============================================================
# 10. ENERJİ ANALİZİ
# ============================================================

hourly_energy = (
    df.groupby("Hour")["started_session_energy_kwh"]
    .agg(["mean", "median", "max"])
    .round(2)
)

print("\n[7] SAATLERE GÖRE ENERJİ DAĞILIMI")
print("-" * 60)
print(hourly_energy)

hourly_energy.to_csv(
    OUTPUT_DIR / "saatlik_enerji_istatistikleri.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(12, 5))

plt.bar(
    hourly_energy.index,
    hourly_energy["mean"]
)

plt.title("Günün Saatlerine Göre Ortalama Oturum Enerjisi")
plt.xlabel("Saat")
plt.ylabel("Ortalama Enerji (kWh)")
plt.xticks(range(24))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_saatlik_enerji.png",
    dpi=300
)
plt.close()

# ============================================================
# 11. GÜNLÜK TOPLAM TALEP
# ============================================================

daily_demand = (
    complete_df.groupby("Date")
    .agg(
        daily_sessions=("session_start_count", "sum"),
        daily_energy=("started_session_energy_kwh", "sum")
    )
)

print("\n[8] TAM GÜNLERİN TALEP İSTATİSTİKLERİ")
print("-" * 60)
print(daily_demand.describe().round(2))

daily_demand.to_csv(
    OUTPUT_DIR / "gunluk_talep_istatistikleri.csv",
    encoding="utf-8-sig"
)

plt.figure(figsize=(14, 5))

plt.plot(
    pd.to_datetime(daily_demand.index),
    daily_demand["daily_sessions"],
    linewidth=0.8
)

plt.title("Günlük Toplam Şarj Oturumu Sayısı")
plt.xlabel("Tarih")
plt.ylabel("Günlük Oturum Sayısı")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "07_gunluk_talep.png",
    dpi=300
)
plt.close()

# ============================================================
# 12. ANALİZ ÖZETİ
# ============================================================

summary = []

summary.append("ELEKTRİKLİ ARAÇ ŞARJ TALEBİ - EDA ÖZETİ")
summary.append("=" * 55)

summary.append(f"Toplam saatlik kayıt: {len(df)}")
summary.append(
    f"Başlangıç: {df['DateTime'].min()}"
)
summary.append(
    f"Bitiş: {df['DateTime'].max()}"
)
summary.append(
    f"Toplam oturum: {df['session_start_count'].sum():.0f}"
)
summary.append(
    f"Toplam oturum enerjisi: "
    f"{df['started_session_energy_kwh'].sum():.2f} kWh"
)
summary.append(
    f"Sıfır oturumlu saat: {zero_hours}"
)
summary.append(
    f"Tam gün sayısı: {len(complete_dates)}"
)

summary.append("\nEn yoğun saatler:")

for hour, value in (
    hourly_demand["mean"]
    .sort_values(ascending=False)
    .head(5)
    .items()
):
    summary.append(
        f"{hour:02d}:00 - {value:.2f} oturum/saat"
    )

summary.append("\nHaftanın günlerine göre talep:")

for day, value in weekday_demand["mean"].items():
    summary.append(f"{day}: {value:.2f}")

summary_text = "\n".join(summary)

with open(
    OUTPUT_DIR / "eda_genel_ozet.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary_text)

print("\n" + "=" * 60)
print("EDA ANALİZİ TAMAMLANDI")
print("=" * 60)

print("Çıktı klasörü:", OUTPUT_DIR)
print("Grafikler ve tablolar kaydedildi.")
print("\n", summary_text)
