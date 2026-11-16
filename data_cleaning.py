
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. DOSYA YOLLARI VE AYARLAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "ChargingRecords.csv"
OUTPUT_DIR = BASE_DIR / "outputs_cleaning"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Duration ile bağlantı süresi arasındaki kabul edilen
# pozitif fark toleransı (dakika)
DURATION_TOLERANCE = 5


# ============================================================
# 2. YARDIMCI FONKSİYONLAR
# ============================================================

def create_datetime(date_series, time_series):
    """Tarih ve saat sütunlarını birleştirerek datetime oluşturur."""

    combined = (
        date_series.astype("string").str.strip()
        + " "
        + time_series.astype("string").str.strip()
    )

    return pd.to_datetime(combined, errors="coerce")


def save_csv(dataframe, filename):
    """CSV dosyasını UTF-8 BOM ile kaydeder."""

    output_path = OUTPUT_DIR / filename
    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Kaydedildi: {output_path.name}")


# ============================================================
# 3. VERİYİ OKUMA
# ============================================================

def run_data_cleaning():

    print("=" * 60)
    print("ELEKTRİKLİ ARAÇ ŞARJ VERİSİ - VERİ HAZIRLAMA")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı:\n{INPUT_FILE}"
        )

    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="cp1254")

    original_columns = df.columns.tolist()
    original_row_count = len(df)

    print(f"\nOrijinal kayıt sayısı: {original_row_count}")
    print(f"Sütun sayısı: {len(original_columns)}")

    # Gerekli sütunların kontrolü
    required_columns = [
        "UserID",
        "ChargerID",
        "ChargerType",
        "StartDay",
        "StartTime",
        "EndDay",
        "EndTime",
        "Duration",
        "Demand"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "CSV dosyasında gerekli sütunlar bulunamadı: "
            + ", ".join(missing_columns)
            + "\nMevcut sütunlar: "
            + ", ".join(df.columns.astype(str))
        )

    # ========================================================
    # 4. TARİH VE SAYISAL VERİLERİN HAZIRLANMASI
    # ========================================================

    df["StartFull"] = create_datetime(
        df["StartDay"],
        df["StartTime"]
    )

    df["EndFull"] = create_datetime(
        df["EndDay"],
        df["EndTime"]
    )

    numeric_columns = [
        "UserID",
        "ChargerID",
        "ChargerType",
        "Duration",
        "Demand"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ========================================================
    # 5. TEKRARLI KAYITLARIN TESPİTİ
    # ========================================================

    # Yalnızca orijinal sütunlar üzerinden kontrol edilir.
    duplicate_mask = df.duplicated(
        subset=original_columns,
        keep=False
    )

    duplicate_records = df.loc[duplicate_mask].copy()

    duplicate_groups = df.duplicated(
        subset=original_columns,
        keep="first"
    ).sum()

    # Her tekrarlı gruptan yalnızca bir kayıt tutulur.
    df = df.drop_duplicates(
        subset=original_columns,
        keep="first"
    ).copy()

    print("\n--- TEKRARLI KAYITLAR ---")
    print(f"Tekrarlı gruplardaki toplam satır: {len(duplicate_records)}")
    print(f"Fazladan kaldırılan satır: {duplicate_groups}")

    save_csv(
        duplicate_records,
        "tekrarli_kayitlar.csv"
    )

    # ========================================================
    # 6. SÜRE VE GÜÇ DEĞİŞKENLERİ
    # ========================================================

    # Bağlantı süresi dakika cinsindendir.
    df["ConnectionMinutes"] = (
        df["EndFull"] - df["StartFull"]
    ).dt.total_seconds() / 60

    # Duration ile bağlantı süresi arasındaki fark.
    df["DurationDifference"] = (
        df["Duration"] - df["ConnectionMinutes"]
    )

    # Yaklaşık ortalama güç (kW).
    # Demand / Duration gerçek anlık şebeke gücü değildir.
    df["ApproxPower_kW"] = np.nan

    valid_duration = df["Duration"] > 0

    df.loc[valid_duration, "ApproxPower_kW"] = (
        df.loc[valid_duration, "Demand"]
        / (df.loc[valid_duration, "Duration"] / 60)
    )

    # ========================================================
    # 7. VERİ KALİTESİ BAYRAKLARI
    # ========================================================

    df["is_anonymous_user"] = df["UserID"].eq(0)

    df["is_negative_duration"] = df["Duration"] < 0
    df["is_zero_duration"] = df["Duration"].eq(0)

    df["is_negative_connection_interval"] = (
        df["ConnectionMinutes"] < 0
    )

    df["is_invalid_start_time"] = df["StartFull"].isna()
    df["is_invalid_end_time"] = df["EndFull"].isna()

    df["is_duration_longer_than_connection"] = (
        df["DurationDifference"] > DURATION_TOLERANCE
    )

    df["is_idle_parking_candidate"] = (
        df["DurationDifference"] < -DURATION_TOLERANCE
    )

    df["is_power_gt_150"] = df["ApproxPower_kW"] > 150
    df["is_power_gt_350"] = df["ApproxPower_kW"] > 350

    df["is_low_power_lt_0_5"] = (
        df["ApproxPower_kW"] < 0.5
    )

    # ChargerType değerleri resmi veri sözlüğü olmadan
    # AC/DC olarak yorumlanmaz. Bu bayraklar yalnızca
    # incelenmesi gereken kayıtları belirler.

    df["is_ct0_power_gt_22"] = (
        df["ChargerType"].eq(0)
        & (df["ApproxPower_kW"] > 22)
    )

    df["is_ct1_power_gt_100"] = (
        df["ChargerType"].eq(1)
        & (df["ApproxPower_kW"] > 100)
    )

    # ========================================================
    # 8. SÜRE VE ZAMAN ANOMALİLERİNİ AYIRMA
    # ========================================================

    anomaly_mask = (
        df["Duration"].le(0)
        | df["ConnectionMinutes"].le(0)
        | df["DurationDifference"].abs().gt(
            DURATION_TOLERANCE
        )
        | df["StartFull"].isna()
        | df["EndFull"].isna()
    )

    anomaly_records = df.loc[anomaly_mask].copy()

    save_csv(
        anomaly_records,
        "sure_ve_zaman_inceleme.csv"
    )

    print("\n--- SÜRE VE ZAMAN İNCELEMESİ ---")
    print(f"İncelenecek kayıt sayısı: {len(anomaly_records)}")

    # ========================================================
    # 9. OTURUM TAHMİNİ VERİSİ
    # ========================================================

    # Oturum başlangıç sayısı tahmininde Duration zorunlu
    # değildir. Bu nedenle sıfır/negatif Duration kayıtları
    # yalnızca bu nedenle silinmez.
    #
    # UserID=0 kayıtları da korunur.

    session_mask = (
        df["StartFull"].notna()
        & df["Demand"].notna()
        & (df["Demand"] >= 0)
    )

    session_data = df.loc[session_mask].copy()

    save_csv(
        session_data,
        "oturum_tahmin_verisi.csv"
    )

    print("\n--- OTURUM TAHMİN VERİSİ ---")
    print(f"Kullanılabilir kayıt: {len(session_data)}")

    # ========================================================
    # 10. SAATLİK OTURUM TAHMİN VERİSİ
    # ========================================================

    session_data["Hour"] = (
        session_data["StartFull"].dt.floor("h")
    )

    hourly_data = (
        session_data
        .groupby("Hour")
        .agg(
            session_start_count=("StartFull", "count"),
            started_session_energy_kwh=("Demand", "sum"),
            anonymous_session_count=("is_anonymous_user", "sum")
        )
    )

    # ChargerType bazında sayımlar
    type_counts = (
        session_data
        .pivot_table(
            index="Hour",
            columns="ChargerType",
            values="StartFull",
            aggfunc="count",
            fill_value=0
        )
    )

    for charger_type in [0, 1]:
        column_name = f"charger_type_{charger_type}_count"

        if charger_type in type_counts.columns:
            hourly_data[column_name] = (
                type_counts[charger_type]
            )
        else:
            hourly_data[column_name] = 0

    # Eksik saatleri sıfır ile tamamla.
    if not hourly_data.empty:

        complete_hours = pd.date_range(
            start=hourly_data.index.min(),
            end=hourly_data.index.max(),
            freq="h"
        )

        hourly_data = hourly_data.reindex(
            complete_hours,
            fill_value=0
        )

        hourly_data.index.name = "Hour"

    hourly_data = hourly_data.reset_index()

    hourly_data["anonymous_session_ratio"] = np.where(
        hourly_data["session_start_count"] > 0,
        hourly_data["anonymous_session_count"]
        / hourly_data["session_start_count"],
        0
    )

    save_csv(
        hourly_data,
        "saatlik_oturum_tahmin_verisi.csv"
    )

    print("\n--- SAATLİK VERİ ---")
    print(f"Saatlik kayıt sayısı: {len(hourly_data)}")

    # ========================================================
    # 11. GÜÇ ANALİZİ VERİSİ
    # ========================================================

    # Güç analizi için geçerli süre ve zaman bilgisi gerekir.
    # Yüksek yaklaşık güç değerleri otomatik silinmez;
    # yalnızca kalite bayraklarıyla işaretlenir.

    power_mask = (
        df["StartFull"].notna()
        & df["EndFull"].notna()
        & (df["Duration"] > 0)
        & (df["Demand"] >= 0)
        & (df["ConnectionMinutes"] > 0)
        & (
            df["DurationDifference"]
            <= DURATION_TOLERANCE
        )
    )

    power_data = df.loc[power_mask].copy()

    save_csv(
        power_data,
        "guc_analizi_verisi.csv"
    )

    print("\n--- GÜÇ ANALİZİ VERİSİ ---")
    print(f"Kullanılabilir kayıt: {len(power_data)}")

    # ========================================================
    # 12. VERİ KALİTESİ ÖZETİ
    # ========================================================

    user_zero_count = df["UserID"].eq(0).sum()

    total_demand = df["Demand"].sum()

    summary = {
        "original_row_count": original_row_count,
        "original_column_count": len(original_columns),

        "remaining_after_duplicate_removal": len(df),
        "duplicate_groups": int(duplicate_groups),
        "duplicate_rows_involved": len(duplicate_records),

        "session_data_row_count": len(session_data),
        "power_data_row_count": len(power_data),
        "hourly_data_row_count": len(hourly_data),

        "negative_duration_count": int(
            df["is_negative_duration"].sum()
        ),

        "zero_duration_count": int(
            df["is_zero_duration"].sum()
        ),

        "negative_connection_interval_count": int(
            df["is_negative_connection_interval"].sum()
        ),

        "duration_longer_than_connection_count": int(
            df["is_duration_longer_than_connection"].sum()
        ),

        "idle_parking_candidate_count": int(
            df["is_idle_parking_candidate"].sum()
        ),

        "invalid_start_time_count": int(
            df["is_invalid_start_time"].sum()
        ),

        "invalid_end_time_count": int(
            df["is_invalid_end_time"].sum()
        ),

        "approx_power_over_150_count": int(
            df["is_power_gt_150"].sum()
        ),

        "approx_power_over_350_count": int(
            df["is_power_gt_350"].sum()
        ),

        "anonymous_user_count": int(user_zero_count),

        "anonymous_user_percentage": (
            round(user_zero_count / len(df) * 100, 2)
            if len(df) > 0 else 0
        ),

        "missing_demand_count": int(
            df["Demand"].isna().sum()
        ),

        "negative_demand_count": int(
            (df["Demand"] < 0).sum()
        ),

        "total_demand_kwh": total_demand,

        "start_date_min": (
            str(df["StartFull"].min())
            if df["StartFull"].notna().any()
            else None
        ),

        "start_date_max": (
            str(df["StartFull"].max())
            if df["StartFull"].notna().any()
            else None
        )
    }

    summary_df = pd.DataFrame(
        list(summary.items()),
        columns=["Metric", "Value"]
    )

    save_csv(
        summary_df,
        "veri_kalite_ozeti.csv"
    )

    # ========================================================
    # 13. SONUÇ RAPORU
    # ========================================================

    print("\n" + "=" * 60)
    print("VERİ HAZIRLAMA TAMAMLANDI")
    print("=" * 60)

    print(f"Orijinal kayıt: {original_row_count}")
    print(f"Tekrarlar çıkarıldıktan sonra: {len(df)}")
    print(f"Oturum tahmini verisi: {len(session_data)}")
    print(f"Güç analizi verisi: {len(power_data)}")
    print(f"Saatlik veri sayısı: {len(hourly_data)}")

    print("\nÇıktı klasörü:")
    print(OUTPUT_DIR)

    print("\nOluşturulan dosyalar:")
    for file in sorted(OUTPUT_DIR.glob("*.csv")):
        print(" -", file.name)

    print("\nOrijinal CSV dosyası değiştirilmedi.")


# ============================================================
# 14. PROGRAMI ÇALIŞTIR
# ============================================================

if __name__ == "__main__":
    run_data_cleaning()
