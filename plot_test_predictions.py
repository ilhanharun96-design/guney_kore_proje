
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --------------------------------------------------
# DOSYA YOLLARI
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "outputs_modeling" / "test_tahminleri.csv"
OUTPUT_DIR = BASE_DIR / "outputs_modeling" / "grafikler"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# VERİYİ OKU
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["target_datetime"] = pd.to_datetime(df["target_datetime"])
df["origin_datetime"] = pd.to_datetime(df["origin_datetime"])

# 24 saatlik tahminleri seç
df24 = df[df["forecast_horizon"] == 24].copy()

if df24.empty:
    raise ValueError("24 saatlik tahmin kaydı bulunamadı.")

# Aynı hedef saat için birden fazla 24 saatlik tahmin var mı?
duplicate_count = df24.duplicated(
    subset=["target", "target_datetime"]
).sum()

if duplicate_count > 0:
    raise ValueError(
        f"24 saatlik tahminlerde {duplicate_count} mükerrer kayıt bulundu."
    )

print("=" * 60)
print("24 SAATLİK TAHMİN GRAFİKLERİ")
print("=" * 60)

print("Toplam 24 saatlik tahmin kaydı:", len(df24))
print("Tahmin dönemi:",
      df24["target_datetime"].min(),
      "-",
      df24["target_datetime"].max())

# --------------------------------------------------
# HER HEDEF İÇİN GRAFİKLER
# --------------------------------------------------

targets = {
    "session_start_count": {
        "title": "Saatlik Şarj Oturumu Sayısı",
        "ylabel": "Oturum Sayısı",
        "unit": "oturum",
        "prefix": "oturum"
    },
    "started_session_energy_kwh": {
        "title": "Başlatılan Oturumlara Atfedilen Enerji",
        "ylabel": "Enerji (kWh)",
        "unit": "kWh",
        "prefix": "enerji"
    }
}

for target, info in targets.items():

    data = df24[df24["target"] == target].copy()
    data = data.sort_values("target_datetime")

    if data.empty:
        print(f"UYARI: {target} için veri bulunamadı.")
        continue

    # Gerçek ve tahmin değerlerini kontrol et
    print("\n" + "-" * 60)
    print(info["title"])
    print("-" * 60)

    print("Kayıt sayısı:", len(data))
    print("Gerçek değer ortalaması:",
          round(data["actual"].mean(), 2))
    print("Tahmin ortalaması:",
          round(data["predicted"].mean(), 2))
    print("Negatif tahmin sayısı:",
          int((data["predicted"] < 0).sum()))

    # --------------------------------------------------
    # 1. ZAMAN SERİSİ GRAFİĞİ
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        data["target_datetime"],
        data["actual"],
        label="Gerçek Değer",
        linewidth=1.5
    )

    ax.plot(
        data["target_datetime"],
        data["predicted"],
        label="24 Saat Önceden Tahmin",
        linewidth=1.5,
        alpha=0.85
    )

    ax.set_title(
        info["title"] + " - Gerçek ve Tahmin (24 Saat Ufuk)",
        fontsize=13
    )
    ax.set_xlabel("Tarih")
    ax.set_ylabel(info["ylabel"])
    ax.legend()
    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = (
        OUTPUT_DIR / f"01_{info['prefix']}_gercek_tahmin_24saat.png"
    )

    plt.savefig(chart_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # 2. GERÇEK - TAHMİN SAÇILIM GRAFİĞİ
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        data["actual"],
        data["predicted"],
        alpha=0.4
    )

    minimum = min(
        data["actual"].min(),
        data["predicted"].min()
    )

    maximum = max(
        data["actual"].max(),
        data["predicted"].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Kusursuz Tahmin Çizgisi"
    )

    ax.set_title(
        info["title"] + " - Gerçek / Tahmin Dağılımı"
    )
    ax.set_xlabel("Gerçek Değer")
    ax.set_ylabel("Tahmin Değeri")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()

    scatter_path = (
        OUTPUT_DIR / f"02_{info['prefix']}_sacilim.png"
    )

    plt.savefig(scatter_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # 3. TAHMİN HATASI ZAMAN SERİSİ
    # --------------------------------------------------

    data["error"] = data["predicted"] - data["actual"]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        data["target_datetime"],
        data["error"],
        linewidth=1
    )

    ax.axhline(0, linestyle="--")

    ax.set_title(
        info["title"] + " - Tahmin Hatası (24 Saat Ufuk)"
    )
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Tahmin - Gerçek")
    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    plt.xticks(rotation=45)
    plt.tight_layout()

    error_path = (
        OUTPUT_DIR / f"03_{info['prefix']}_hata_zaman_serisi.png"
    )

    plt.savefig(error_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # 4. SAAT BAZINDA HATA ÖZETİ
    # --------------------------------------------------

    data["hour"] = data["target_datetime"].dt.hour
    data["absolute_error"] = data["error"].abs()

    hourly_error = (
        data.groupby("hour")
        .agg(
            MAE=("absolute_error", "mean"),
            Ortalama_Gercek=("actual", "mean"),
            Ortalama_Tahmin=("predicted", "mean")
        )
        .reset_index()
    )

    hourly_error.to_csv(
        OUTPUT_DIR / f"04_{info['prefix']}_saatlik_hata.csv",
        index=False
    )

    print("\nSaat bazında MAE:")
    print(hourly_error.round(2).to_string(index=False))

    print("\nGrafikler kaydedildi.")

print("\n" + "=" * 60)
print("TÜM GRAFİKLER OLUŞTURULDU.")
print("Kayıt klasörü:", OUTPUT_DIR)
print("=" * 60)
