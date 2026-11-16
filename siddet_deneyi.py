"""
siddet_deneyi.py
================
A Aşaması — Günlük Aşım Enerjisi Şiddet Deneyi (Birebir İndeks Doğrulamalı Mimari)

Metodoloji:
  - Tahmin zamanı: Her hedef gün d için origin = (d-1) 23:00
  - Hedef pencere: d 00:00 – d 23:00 (forecast_horizon = 1..24, tam 24 saatlik kesintisiz dizi)
  - Kapasite referansı: Kıvrım bazlı (her kıvrımın bilgi kesme tarihine kadar %99 persentil)
  - B0 = profil(wd, hr) / profil_genel_ort * rolling_mean_168 @ origin
  - S1 = saatlik ihtiyatlı üst sınır (karekök ölçek, q=0.90 kalibrasyon artığı)
        NOT: Günlük %90 güven sınırı DEĞİLDİR; saatlik banttan türetilen ihtiyatlı skor.
  - Persistans-168 ve Profil-statik aynı origin bilgi kümesiyle oluşturulur.
  - Başarı ölçütleri önceden kilitlendi (S1, S2, S3); gevşetilemez.
  - Birebir İndeks ve Savunmacı Mimari Kuralları:
      * Yük serisi indeksi beklenen saatlik dizi ile birebir eşleşmelidir (eksik ve fazla saat denetimi).
      * Ön kontrollerde hata varsa ana döngüye girilmez (KeyError/reindex koruması).
      * Sonradan veri hatası tespit edilirse kısmi hesaplanan sonuçlar geçersiz kılınır.
      * Veri hatasında siddet_gun_detay.csv ve siddet_korelasyonlar.csv açıkça
        NOT EVALUABLE durumunu gösterir; hiçbir PASS/FAIL üretilmez.
      * Eksik günler ve hata nedenleri özet rapora satır satır yazdırılır.
  - Tanımsızlık Kuralı:
      * Herhangi bir kıvrımda veya havuzda B0 Spearman tanımsızsa -> S2 = NOT EVALUABLE.
      * Baseline'lardan herhangi biri tanımsızsa -> S1 = NOT EVALUABLE.
      * NaN değerler sıfırla doldurulamaz, ortalamadan sessizce çıkarılamaz.
  - 3 kıvrım aritmetik ortalaması yalnızca üçü de tanımlıysa hesaplanır.
  - Eylül 2022 test verisi assert ile doğrulanır, kesinlikle kullanılmaz.

Girdi dosyaları (salt okunur):
  - outputs_feature_engineering/train_verisi.csv
  - outputs_feature_engineering/validation_verisi.csv
  - outputs_load_profile/saatlik_yaklasik_yuk.csv

Çıktılar (outputs_load_profile/siddet_deneyi/):
  - siddet_gun_detay.csv       : Gün × senaryo × kapasite düzeyinde aşım enerjileri
  - siddet_korelasyonlar.csv   : Kıvrım bazlı ve havuzlanmış Spearman / Pearson
  - siddet_ozeti.txt           : Okunabilir rapor ve PASS/FAIL/NOT EVALUABLE kararları
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# ============================================================
# YOL AYARLARI
# ============================================================
BASE = Path(__file__).resolve().parent
FE   = BASE / "outputs_feature_engineering"
LP   = BASE / "outputs_load_profile"
OUT  = LP / "siddet_deneyi"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SABİTLER (kilitlendi; değiştirilemez)
# ============================================================
TEST_BAS           = pd.Timestamp("2022-09-01")
PROFIL_BAS         = pd.Timestamp("2021-10-07 18:00")
YUK_BITIS_BEKLENEN = pd.Timestamp("2022-08-31 23:00")
BOSLUK             = pd.Timedelta(days=2)        # Profil / kalibrasyon kesme tamponu
KAL_HAFTA          = 8                           # Kalibrasyon penceresi (hafta)
Q_UPPER            = 0.90                        # S1 saatlik üst sınır kalibrasyon yüzdesi
EPS_SCALE          = 0.05                        # Ölçek kararlılık sabiti (B0 ~0 için)
KAPASITE_ORANLARI  = [0.70, 0.80, 0.90]
SERILER            = {                           # Ana: U100; duyarlılık: F50, F25
    "U100": "yuk_U100_kwh",
    "F50":  "yuk_F50_kwh",
    "F25":  "yuk_F25_kwh",
}
SEVIYE_KOL         = "started_session_energy_kwh_rolling_mean_168"
FOLDLAR = [
    ("2022-06", pd.Timestamp("2022-06-01"), pd.Timestamp("2022-07-01"), 30),
    ("2022-07", pd.Timestamp("2022-07-01"), pd.Timestamp("2022-08-01"), 31),
    ("2022-08", pd.Timestamp("2022-08-01"), pd.Timestamp("2022-09-01"), 31),
]
BEKLENEN_TOPLAM_GUN = sum(f[3] for f in FOLDLAR)  # 30 + 31 + 31 = 92 gün

# Başarı ölçütleri (kilitlendi)
S2_HAVUZ_ESIK   = 0.60
S2_KIVRIM_MIN   = 0.45
S3_HAVUZ_ESIK   = 0.50
KARAR_SC        = "U100"
KARAR_KAP       = [0.70, 0.80]

SATIRLAR = []

def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))

def bolum(baslik):
    yaz()
    yaz("=" * 76)
    yaz(baslik)
    yaz("=" * 76)

# ============================================================
# İSTATİSTİK FONKSİYONLARI (KATI KONTROLLÜ)
# ============================================================

def spearman_guvende(gercek, tahmin):
    """
    Spearman sıra korelasyonu.
    Ayrı ayrı kontroller:
      - None kontrolü
      - Boş dizi kontrolü (len == 0)
      - Uzunluk eşitliği ve asgari eleman sayısı (len >= 2)
      - NaN / Inf / eksik değer kontrolü
      - Sıfır varyans kontrolü (gerçek veya tahmin sabit seri)
      - Sonlu hesaplama sonucu kontrolü
    Herhangi bir ihlalde (np.nan, False) döner.
    """
    if gercek is None or tahmin is None:
        return np.nan, False
    g = np.asarray(gercek, dtype=float)
    t = np.asarray(tahmin, dtype=float)
    if len(g) == 0 or len(t) == 0:
        return np.nan, False
    if len(g) != len(t):
        return np.nan, False
    if len(g) < 2:
        return np.nan, False
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(t)):
        return np.nan, False
    if float(np.std(g)) == 0.0 or float(np.std(t)) == 0.0:
        return np.nan, False
    res = spearmanr(g, t)
    rho = float(res.statistic if hasattr(res, "statistic") else res[0])
    if not np.isfinite(rho):
        return np.nan, False
    return rho, True


def pearson_guvende(gercek, tahmin):
    """Pearson korelasyonu (katı sonluluk ve varyans kontrollü)."""
    if gercek is None or tahmin is None:
        return np.nan
    g = np.asarray(gercek, dtype=float)
    t = np.asarray(tahmin, dtype=float)
    if len(g) == 0 or len(t) == 0 or len(g) != len(t) or len(g) < 2:
        return np.nan
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(t)):
        return np.nan
    if float(np.std(g)) == 0.0 or float(np.std(t)) == 0.0:
        return np.nan
    res = pearsonr(g, t)
    r = float(res.statistic if hasattr(res, "statistic") else res[0])
    if not np.isfinite(r):
        return np.nan
    return r


def profil_hesapla(yuk_seri, bitis):
    """
    [PROFIL_BAS, bitis) aralığındaki yük serisinden
    hafta-gün x saat ortalama profili ve genel ortalaması.
    """
    s = yuk_seri[(yuk_seri.index >= PROFIL_BAS) & (yuk_seri.index < bitis)]
    if len(s) == 0:
        raise ValueError(f"Profil için veri yok: PROFIL_BAS={PROFIL_BAS}, bitis={bitis}")
    df_p = pd.DataFrame({
        "v":  s.values,
        "wd": s.index.dayofweek,
        "hr": s.index.hour,
    })
    prof_df   = (df_p.groupby(["wd", "hr"])["v"]
                 .mean().reset_index().rename(columns={"v": "profil"}))
    genel_ort = float(s.mean())
    return prof_df, genel_ort


def b0_vektorel(fe_df, prof_df, genel_ort):
    """B0 nokta tahmini (vektörel)."""
    merged = fe_df[["target_weekday", "target_hour", SEVIYE_KOL]].merge(
        prof_df,
        left_on=["target_weekday", "target_hour"],
        right_on=["wd", "hr"],
        how="left",
    )
    profil_vals = merged["profil"].fillna(genel_ort).values
    seviye_vals = fe_df[SEVIYE_KOL].values
    if genel_ort == 0:
        return np.zeros(len(fe_df), dtype=float)
    return np.clip(profil_vals / genel_ort * seviye_vals, 0.0, None)


def profil_statik_vektorel(fe_df, prof_df, genel_ort):
    """Statik profil tahmini (seviye ölçeklemesi yok)."""
    merged = fe_df[["target_weekday", "target_hour", SEVIYE_KOL]].merge(
        prof_df,
        left_on=["target_weekday", "target_hour"],
        right_on=["wd", "hr"],
        how="left",
    )
    return np.clip(merged["profil"].fillna(genel_ort).values, 0.0, None)


def asim_enerjisi(degerler, kapasite):
    """Günlük aşım enerjisi: sum_t max(0, v_t - C) [kWh]"""
    return float(np.maximum(0.0, np.asarray(degerler, dtype=float) - kapasite).sum())

# ============================================================
# BÖLÜM 0: VERİ YÜKLEME VE KATI ÖN KONTROLLER (GATEKEEPER)
# ============================================================
bolum("0. VERİ YÜKLEME VE BİREBİR İNDEKS KONTROLLERİ")

veri_eksik = False
on_kontrol_hatalari = []
eksik_gunler = []

# 1. Özellik Verisi Yükleme ve Eylül 2022 Sızıntı Denetimi
FE_COLS = [
    "origin_datetime", "target_datetime", "forecast_horizon",
    "target_weekday",  "target_hour",
    SEVIYE_KOL,
]

train = pd.read_csv(
    FE / "train_verisi.csv",
    usecols=FE_COLS,
    parse_dates=["origin_datetime", "target_datetime"],
    encoding="utf-8-sig",
)
val = pd.read_csv(
    FE / "validation_verisi.csv",
    usecols=FE_COLS,
    parse_dates=["origin_datetime", "target_datetime"],
    encoding="utf-8-sig",
)
tum_fe = pd.concat([train, val], ignore_index=True)

assert tum_fe["target_datetime"].max() < TEST_BAS, \
    "HATA: Özellik verisinde Eylül 2022 (test) target_datetime var!"
assert tum_fe["origin_datetime"].max() < TEST_BAS, \
    "HATA: Özellik verisinde Eylül 2022 origin_datetime var!"
yaz("  [GEÇTİ] Özellik verisinde Eylül 2022 yok.")
yaz(f"  Özellik verisi: {tum_fe['target_datetime'].min()} - "
    f"{tum_fe['target_datetime'].max()}, {len(tum_fe):,} satır")

# 2. Yük Verisi Yükleme
yuk_df = pd.read_csv(
    LP / "saatlik_yaklasik_yuk.csv",
    parse_dates=["datetime"],
    encoding="utf-8-sig",
)
yuk_df = yuk_df.set_index("datetime").sort_index()
yuk_df = yuk_df[yuk_df.index < TEST_BAS]

assert yuk_df.index.max() < TEST_BAS, \
    "HATA: Yük serisinde Eylül 2022 (test) var!"
yaz("  [GEÇTİ] Yük serisinde Eylül 2022 yok.")

# Analiz dönemi yük serisi (PROFIL_BAS ile YUK_BITIS_BEKLENEN arası)
yuk_df_analiz = yuk_df[(yuk_df.index >= PROFIL_BAS) & (yuk_df.index <= YUK_BITIS_BEKLENEN)].copy()

# 3. Yük Serisi Birebir İndeks Bütünlüğü Denetimleri
if yuk_df_analiz.index.has_duplicates:
    hata_msg = "Analiz dönemi yük serisinde tekrarlayan zaman damgaları bulundu!"
    yaz(f"  HATA: {hata_msg}")
    on_kontrol_hatalari.append(hata_msg)
    veri_eksik = True
else:
    yaz("  [GEÇTİ] Yük serisi indeksinde tekrarlı kayıt bulunmuyor.")

# Sütun varlığı kontrolü
eksik_sutunlar = [kol for sc, kol in SERILER.items() if kol not in yuk_df_analiz.columns]
if eksik_sutunlar:
    hata_msg = f"Yük tablosunda gerekli sütunlar eksik: {eksik_sutunlar}"
    yaz(f"  HATA: {hata_msg}")
    on_kontrol_hatalari.append(hata_msg)
    veri_eksik = True
else:
    yaz("  [GEÇTİ] Gerekli tüm yük senaryosu sütunları mevcut.")

# Birebir saatlik zaman dizisi denetimi (Eksik ve Fazla saatler)
beklenen_yuk_index = pd.date_range(PROFIL_BAS, YUK_BITIS_BEKLENEN, freq="h")
eksik_saatler_yuk = beklenen_yuk_index.difference(yuk_df_analiz.index)
fazla_saatler_yuk = yuk_df_analiz.index.difference(beklenen_yuk_index)

if len(eksik_saatler_yuk) > 0:
    ornekler = [str(ts) for ts in eksik_saatler_yuk[:3]]
    hata_msg = f"Yük serisinde {len(eksik_saatler_yuk)} eksik saat tespit edildi! Örnekler: {ornekler}"
    yaz(f"  HATA: {hata_msg}")
    on_kontrol_hatalari.append(hata_msg)
    veri_eksik = True

if len(fazla_saatler_yuk) > 0:
    ornekler = [str(ts) for ts in fazla_saatler_yuk[:3]]
    hata_msg = f"Yük serisinde beklenen dizinin dışında kalan {len(fazla_saatler_yuk)} beklenmeyen/fazla zaman damgası tespit edildi! Örnekler: {ornekler}"
    yaz(f"  HATA: {hata_msg}")
    on_kontrol_hatalari.append(hata_msg)
    veri_eksik = True

if not yuk_df_analiz.index.equals(beklenen_yuk_index):
    hata_msg = "Yük serisi zaman indeksi beklenen saatlik dizi ile birebir eşleşmiyor!"
    if not (len(eksik_saatler_yuk) > 0 or len(fazla_saatler_yuk) > 0):
        yaz(f"  HATA: {hata_msg}")
        on_kontrol_hatalari.append(hata_msg)
    veri_eksik = True
else:
    yaz(f"  [GEÇTİ] Yük serisi zaman indeksi beklenen saatlik dizi ile birebir eşleşiyor ({len(beklenen_yuk_index)} saat).")

# Sonlu değer kontrolü
if not yuk_df_analiz.index.has_duplicates and not eksik_sutunlar and not veri_eksik:
    for sc, kol in SERILER.items():
        seri_vals = yuk_df_analiz[kol].values
        if not np.all(np.isfinite(seri_vals)):
            hata_msg = f"Yük serisinde sonlu olmayan (NaN/Inf) değer var: {sc}"
            yaz(f"  HATA: {hata_msg}")
            on_kontrol_hatalari.append(hata_msg)
            veri_eksik = True
    if not veri_eksik:
        yaz("  [GEÇTİ] Tüm senaryolara ait yük serisi değerleri sonlu ve geçerli.")

# ============================================================
# BÖLÜM 1: ANA HESAPLAMA DÖNGÜSÜ
# ============================================================
bolum("1. GÜNLÜK AŞIM ENERJİSİ HESABI VE VERİ BÜTÜNLÜĞÜ")
yaz("Not: Hedef gün d için origin = (d-1) 23:00.")
yaz("Not: S1 = saatlik ihtiyatlı üst sınırdan türetilen günlük aşım skoru;")
yaz("     günlük %90 güven sınırı DEĞİLDİR.")

gun_kayit = []

if veri_eksik:
    yaz("\nKRİTİK UYARI: Ön kontrollerde veri hatası tespit edildiği için ana hesaplama döngüsü ATLANDI.")
    yaz("Tekrarlı veya eksik indeks üzerinde reindex/hesaplama yapılmamıştır.")
else:
    for sc, kolon in SERILER.items():
        yuk_seri = yuk_df_analiz[kolon]
        yaz(f"\n{'='*50}\nSenaryo: {sc}")

        for fold_ad, fold_bas, fold_bit, beklenen_gun_sayisi in FOLDLAR:
            kesme   = fold_bas - BOSLUK
            cal_bas = kesme - pd.Timedelta(weeks=KAL_HAFTA)

            # Kıvrım bazlı %99 referans
            ref_seri = yuk_seri[(yuk_seri.index >= PROFIL_BAS) & (yuk_seri.index < kesme)]
            cap_ref  = float(ref_seri.quantile(0.99))
            yaz(f"\n  Kıvrım {fold_ad}: kesme={kesme.date()}, cap_ref(%99)={cap_ref:.1f} kW ({len(ref_seri)} saat)")

            # Profil
            prof_df, genel_ort = profil_hesapla(yuk_seri, kesme)

            # Kalibrasyon denetimi
            cal_fe = tum_fe[
                (tum_fe["target_datetime"] >= cal_bas) &
                (tum_fe["target_datetime"] <  kesme)
            ].copy()

            if len(cal_fe) == 0:
                yaz(f"  HATA: Kalibrasyon verisi boş: {sc}/{fold_ad}")
                veri_eksik = True
                break

            a_cal = yuk_seri.reindex(cal_fe["target_datetime"]).values
            b_cal = b0_vektorel(cal_fe, prof_df, genel_ort)

            if not np.all(np.isfinite(a_cal)) or not np.all(np.isfinite(b_cal)):
                yaz(f"  HATA: Kalibrasyon yükünde veya tahmininde sonlu olmayan (NaN/Inf) değer var: {sc}/{fold_ad}")
                veri_eksik = True
                break

            m_cal  = float(np.mean(a_cal))
            ol_cal = np.sqrt(np.clip(b_cal + EPS_SCALE * m_cal, 0.0, None))
            masque = np.isfinite(ol_cal) & (ol_cal > 0.0)

            if masque.sum() == 0:
                yaz(f"  HATA: Kalibrasyon ölçek maskesi tamamen boş: {sc}/{fold_ad}")
                veri_eksik = True
                break

            z_cal = np.sort((a_cal[masque] - b_cal[masque]) / ol_cal[masque])
            if len(z_cal) == 0:
                yaz(f"  HATA: Kalibrasyon artık dizisi z_cal boş: {sc}/{fold_ad}")
                veri_eksik = True
                break

            q90 = float(np.quantile(z_cal, Q_UPPER))
            if not np.isfinite(q90):
                yaz(f"  HATA: Kalibrasyon q90 sonlu bir sayı değil: {sc}/{fold_ad}")
                veri_eksik = True
                break

            # Hedef günler
            hedef_gunler = pd.date_range(fold_bas, fold_bit - pd.Timedelta(days=1), freq="D")
            if len(hedef_gunler) != beklenen_gun_sayisi:
                yaz(f"  HATA: Kıvrım {fold_ad} gün sayısı ({len(hedef_gunler)}) beklenenle ({beklenen_gun_sayisi}) uyuşmuyor!")
                veri_eksik = True
                break

            for gun in hedef_gunler:
                origin      = gun - pd.Timedelta(hours=1)
                hedef_bitis = gun + pd.Timedelta(days=1)

                fe_gun = tum_fe[
                    (tum_fe["origin_datetime"] == origin) &
                    (tum_fe["target_datetime"] >= gun) &
                    (tum_fe["target_datetime"] <  hedef_bitis)
                ].sort_values("forecast_horizon").reset_index(drop=True)

                # Satır sayısı ve forecast_horizon tam olarak 1..24 olmalı
                h_beklenen = list(range(1, 25))
                if len(fe_gun) != 24 or fe_gun["forecast_horizon"].tolist() != h_beklenen:
                    yaz(f"  HATA: {sc}/{fold_ad}/{gun.date()} forecast_horizon 1..24 eksik/hatalı!")
                    eksik_gunler.append({"sc": sc, "fold": fold_ad, "gun": str(gun.date()), "neden": "horizon_hatasi"})
                    veri_eksik = True
                    continue

                # Hedef zaman damgalarını tam olarak date_range(freq='h') ile karşılaştır
                hedef_dt = pd.DatetimeIndex(fe_gun["target_datetime"])
                beklenen_hedef_dt = pd.date_range(gun, periods=24, freq="h")
                if not hedef_dt.equals(beklenen_hedef_dt):
                    yaz(f"  HATA: {sc}/{fold_ad}/{gun.date()} hedef zaman damgaları beklenen saatlik seriye tam eşit değil!")
                    eksik_gunler.append({"sc": sc, "fold": fold_ad, "gun": str(gun.date()), "neden": "zaman_damgasi_hatasi"})
                    veri_eksik = True
                    continue

                # target_weekday ve target_hour çapraz kontrolü
                wd_gercek = hedef_dt.dayofweek.values
                hr_gercek = hedef_dt.hour.values
                if not np.array_equal(fe_gun["target_weekday"].values, wd_gercek) or \
                   not np.array_equal(fe_gun["target_hour"].values, hr_gercek):
                    yaz(f"  HATA: {sc}/{fold_ad}/{gun.date()} target_weekday veya target_hour hedef tarihle tutarsız!")
                    eksik_gunler.append({"sc": sc, "fold": fold_ad, "gun": str(gun.date()), "neden": "takvim_ozellik_uyumsuzlugu"})
                    veri_eksik = True
                    continue

                # rolling_mean_168 sonlu, geçerli ve 24 satırda birbirine eşit olmalı
                seviyeler = fe_gun[SEVIYE_KOL].values
                if not np.all(np.isfinite(seviyeler)) or len(np.unique(seviyeler)) != 1:
                    yaz(f"  HATA: {sc}/{fold_ad}/{gun.date()} seviye (rolling_mean_168) sonlu değil veya satırlar arası tutarsız!")
                    eksik_gunler.append({"sc": sc, "fold": fold_ad, "gun": str(gun.date()), "neden": "seviye_tutarsiz"})
                    veri_eksik = True
                    continue

                # Gerçek yük ve Persistans yükü
                gercek = yuk_seri.reindex(hedef_dt).values
                pers_dt = pd.DatetimeIndex(hedef_dt - pd.Timedelta(hours=168))
                pers_preds = yuk_seri.reindex(pers_dt).values

                if not np.all(np.isfinite(gercek)) or not np.all(np.isfinite(pers_preds)):
                    yaz(f"  HATA: {sc}/{fold_ad}/{gun.date()} gerçek yük veya persistans yükünde sonlu olmayan değer var!")
                    eksik_gunler.append({"sc": sc, "fold": fold_ad, "gun": str(gun.date()), "neden": "sonlu_olmayan_yuk"})
                    veri_eksik = True
                    continue

                # Model tahminleri
                b0_preds = b0_vektorel(fe_gun, prof_df, genel_ort)
                ol_eval  = np.sqrt(np.clip(b0_preds + EPS_SCALE * m_cal, 0.0, None))
                s1_preds = np.clip(b0_preds + q90 * ol_eval, 0.0, None)
                profil_preds = profil_statik_vektorel(fe_gun, prof_df, genel_ort)

                # Aşım enerjisi hesabı
                for kap in KAPASITE_ORANLARI:
                    C = kap * cap_ref
                    gun_kayit.append({
                        "senaryo":              sc,
                        "fold":                 fold_ad,
                        "gun":                  str(gun.date()),
                        "kapasite_orani":       kap,
                        "kapasite_kw":          round(C, 2),
                        "E_gercek":             asim_enerjisi(gercek,       C),
                        "E_B0":                 asim_enerjisi(b0_preds,     C),
                        "E_S1":                 asim_enerjisi(s1_preds,     C),
                        "E_Profil":             asim_enerjisi(profil_preds, C),
                        "E_Pers":               asim_enerjisi(pers_preds,   C),
                    })

        if veri_eksik:
            break

# Günlük Detay CSV Dosyasının Güvenli Oluşturulması
beklenen_satir = len(SERILER) * len(KAPASITE_ORANLARI) * BEKLENEN_TOPLAM_GUN

if veri_eksik or len(gun_kayit) != beklenen_satir:
    veri_eksik = True
    yaz(f"\nKRİTİK UYARI: Veri hatası veya eksikliği tespit edildi!")
    yaz("Kısmi hesaplanan günlük aşım sonuçları GEÇERSİZ KILINDI ve normal sonuç olarak sunulmadı.")

    if on_kontrol_hatalari:
        yaz("\nÖN KONTROL HATA DETAYLARI:")
        for h in on_kontrol_hatalari:
            yaz(f"  - {h}")

    if eksik_gunler:
        yaz("\nEKSİK / HATALI GÜNLER LİSTESİ:")
        for eg in eksik_gunler:
            yaz(f"  Senaryo: {eg['sc']} | Kıvrım: {eg['fold']} | Gün: {eg['gun']} | Neden: {eg['neden']}")

    gun_df = pd.DataFrame([{
        "durum": "NOT_EVALUABLE_VERI_HATASI",
        "aciklama": "Veri hatasi veya eksiklik tespit edildigi icin kismi sonuclar gecersiz kilindi.",
        "on_kontrol_hata_sayisi": len(on_kontrol_hatalari),
        "eksik_gun_sayisi": len(eksik_gunler),
    }])
else:
    gun_df = pd.DataFrame(gun_kayit)
    yaz(f"\n[GEÇTİ] Tüm kıvrımlardaki {BEKLENEN_TOPLAM_GUN} gün ve saat damgası tüm kontrollerden eksiksiz geçti.")

gun_df.to_csv(OUT / "siddet_gun_detay.csv", index=False, encoding="utf-8-sig")

# ============================================================
# BÖLÜM 2: KORELASYON HESABI
# ============================================================
bolum("2. KORELASYON HESABI (KIVRIM BAZLI VE HAVUZLANMIŞ)")

kor_kayit = []

if veri_eksik:
    yaz("\nKRİTİK UYARI: Veri eksikliği/hatası nedeniyle korelasyon hesaplamaları İPTAL EDİLMİŞTİR.")
    yaz("Kısmi veya eksik günlerden korelasyon katsayısı üretilmemiştir (Tümü NaN / NOT EVALUABLE).")

    for sc in SERILER:
        for kap in KAPASITE_ORANLARI:
            for fold_ad, _, _, _ in FOLDLAR:
                kor_kayit.append({
                    "senaryo":                sc,
                    "kapasite_orani":         kap,
                    "fold":                   fold_ad,
                    "havuzlanmis":            False,
                    "n_gun":                  0,
                    "n_sifir_gercek":         0,
                    "n_pozitif_gercek":       0,
                    "n_benzersiz_gercek":     0,
                    "spearman_B0_tanimli":    False,
                    "spearman_B0":            np.nan,
                    "spearman_S1":            np.nan,
                    "spearman_Pers_tanimli":  False,
                    "spearman_Pers":          np.nan,
                    "spearman_Profil_tanimli":False,
                    "spearman_Profil":        np.nan,
                    "pearson_B0":             np.nan,
                    "ort_spearman_B0_3kivrim": np.nan,
                    "durum":                  "NOT_EVALUABLE_VERI_HATASI",
                })
            kor_kayit.append({
                "senaryo":                sc,
                "kapasite_orani":         kap,
                "fold":                   "HAVUZ",
                "havuzlanmis":            True,
                "n_gun":                  0,
                "n_sifir_gercek":         0,
                "n_pozitif_gercek":       0,
                "n_benzersiz_gercek":     0,
                "spearman_B0_tanimli":    False,
                "spearman_B0":            np.nan,
                "spearman_S1":            np.nan,
                "spearman_Pers_tanimli":  False,
                "spearman_Pers":          np.nan,
                "spearman_Profil_tanimli":False,
                "spearman_Profil":        np.nan,
                "pearson_B0":             np.nan,
                "ort_spearman_B0_3kivrim": np.nan,
                "durum":                  "NOT_EVALUABLE_VERI_HATASI",
            })
else:
    for sc in SERILER:
        for kap in KAPASITE_ORANLARI:
            alt = gun_df[(gun_df["senaryo"] == sc) & (gun_df["kapasite_orani"] == kap)]

            kivrim_b0_deger   = {}
            kivrim_b0_tanimli = {}

            # Kıvrım bazlı korelasyonlar
            for fold_ad, _, _, beklenen_n in FOLDLAR:
                f = alt[alt["fold"] == fold_ad]
                n_gun_       = len(f)
                n_sifir_     = int((f["E_gercek"] == 0).sum()) if n_gun_ > 0 else 0
                n_pozitif_   = n_gun_ - n_sifir_
                n_benzersiz_ = int(f["E_gercek"].nunique()) if n_gun_ > 0 else 0

                rho_b0,  tanimli_b0  = spearman_guvende(f["E_gercek"], f["E_B0"])
                rho_s1,  tanimli_s1  = spearman_guvende(f["E_gercek"], f["E_S1"])
                rho_prs, tanimli_prs = spearman_guvende(f["E_gercek"], f["E_Pers"])
                rho_prf, tanimli_prf = spearman_guvende(f["E_gercek"], f["E_Profil"])
                prs_b0               = pearson_guvende(f["E_gercek"],  f["E_B0"])

                kivrim_b0_deger[fold_ad]   = rho_b0
                kivrim_b0_tanimli[fold_ad] = tanimli_b0

                kor_kayit.append({
                    "senaryo":                sc,
                    "kapasite_orani":         kap,
                    "fold":                   fold_ad,
                    "havuzlanmis":            False,
                    "n_gun":                  n_gun_,
                    "n_sifir_gercek":         n_sifir_,
                    "n_pozitif_gercek":       n_pozitif_,
                    "n_benzersiz_gercek":     n_benzersiz_,
                    "spearman_B0_tanimli":    tanimli_b0,
                    "spearman_B0":            rho_b0,
                    "spearman_S1":            rho_s1,
                    "spearman_Pers_tanimli":  tanimli_prs,
                    "spearman_Pers":          rho_prs,
                    "spearman_Profil_tanimli":tanimli_prf,
                    "spearman_Profil":        rho_prf,
                    "pearson_B0":             prs_b0,
                    "ort_spearman_B0_3kivrim": np.nan,
                    "durum":                  "HESAPLANDI",
                })

            # Havuzlanmış korelasyonlar
            n_hav         = len(alt)
            n_sifir_hav   = int((alt["E_gercek"] == 0).sum()) if n_hav > 0 else 0
            n_pozitif_hav = n_hav - n_sifir_hav
            n_ben_hav     = int(alt["E_gercek"].nunique()) if n_hav > 0 else 0

            rho_b0_hav,  tanimli_b0_hav  = spearman_guvende(alt["E_gercek"], alt["E_B0"])
            rho_s1_hav,  tanimli_s1_hav  = spearman_guvende(alt["E_gercek"], alt["E_S1"])
            rho_prs_hav, tanimli_prs_hav = spearman_guvende(alt["E_gercek"], alt["E_Pers"])
            rho_prf_hav, tanimli_prf_hav = spearman_guvende(alt["E_gercek"], alt["E_Profil"])
            prs_b0_hav                   = pearson_guvende(alt["E_gercek"],  alt["E_B0"])

            tum_tanimli  = (len(kivrim_b0_tanimli) == len(FOLDLAR)) and all(kivrim_b0_tanimli[f[0]] for f in FOLDLAR)
            ort_spear_b0 = (
                float(np.mean([kivrim_b0_deger[f[0]] for f in FOLDLAR]))
                if tum_tanimli else np.nan
            )

            kor_kayit.append({
                "senaryo":                sc,
                "kapasite_orani":         kap,
                "fold":                   "HAVUZ",
                "havuzlanmis":            True,
                "n_gun":                  n_hav,
                "n_sifir_gercek":         n_sifir_hav,
                "n_pozitif_gercek":       n_pozitif_hav,
                "n_benzersiz_gercek":     n_ben_hav,
                "spearman_B0_tanimli":    tanimli_b0_hav,
                "spearman_B0":            rho_b0_hav,
                "spearman_S1":            rho_s1_hav,
                "spearman_Pers_tanimli":  tanimli_prs_hav,
                "spearman_Pers":          rho_prs_hav,
                "spearman_Profil_tanimli":tanimli_prf_hav,
                "spearman_Profil":        rho_prf_hav,
                "pearson_B0":             prs_b0_hav,
                "ort_spearman_B0_3kivrim": ort_spear_b0,
                "durum":                  "HESAPLANDI",
            })

kor_df = pd.DataFrame(kor_kayit)
kor_df.to_csv(OUT / "siddet_korelasyonlar.csv", index=False, encoding="utf-8-sig")

# Ekrana özet tablolar
bolum("2a. KORELASYON ÖZET TABLOSU (Havuzlanmış)")
if veri_eksik:
    yaz("Veri eksikliği/hatası nedeniyle korelasyon tablosu oluşturulmadı (Tümü NOT EVALUABLE).")
else:
    for sc in SERILER:
        yaz(f"\n[{sc}]")
        hav = kor_df[(kor_df["senaryo"] == sc) & (kor_df["fold"] == "HAVUZ")][
            ["kapasite_orani", "n_gun", "n_sifir_gercek", "n_benzersiz_gercek",
             "spearman_B0_tanimli", "spearman_B0", "spearman_Pers", "spearman_Profil",
             "pearson_B0", "ort_spearman_B0_3kivrim"]
        ]
        yaz(hav.round(3).to_string(index=False))

bolum("2b. KORELASYON KIVRIM BAZLI DETAY")
if veri_eksik:
    yaz("Veri eksikliği/hatası nedeniyle kıvrım detay tablosu oluşturulmadı (Tümü NOT EVALUABLE).")
else:
    for sc in SERILER:
        for kap in KAPASITE_ORANLARI:
            yaz(f"\n[{sc}] Kapasite %{int(kap*100)}")
            detay = kor_df[
                (kor_df["senaryo"] == sc) &
                (kor_df["kapasite_orani"] == kap)
            ][["fold", "n_gun", "n_sifir_gercek", "n_pozitif_gercek",
               "spearman_B0_tanimli", "spearman_B0", "spearman_Pers",
               "spearman_Profil", "pearson_B0"]]
            yaz(detay.round(3).to_string(index=False))

# ============================================================
# BÖLÜM 3: BAŞARI ÖLÇÜTLERİ DEĞERLENDİRMESİ
# ============================================================
bolum("3. BAŞARI ÖLÇÜTLERİ (S1, S2, S3) - ÖNCEDEN KİLİTLENMİŞTİR")
yaz(f"Karar senaryosu : {KARAR_SC}")
yaz(f"Karar kapasite  : {[f'%{int(k*100)}' for k in KARAR_KAP]}")
yaz("S1              : Havuzlanmış Spearman(B0) > max(Spearman(Pers), Spearman(Profil))")
yaz("                  (Her iki baseline da tanımlı olmalıdır; aksi halde NOT EVALUABLE)")
yaz(f"S2              : Havuzlanmış Spearman(B0) >= {S2_HAVUZ_ESIK}")
yaz(f"                  VE hiçbir kıvrımda Spearman(B0) < {S2_KIVRIM_MIN}")
yaz("                  (Havuz veya herhangi bir kıvrım tanımsız ise NOT EVALUABLE)")
yaz(f"S3 (bilgi)      : {KARAR_SC} %90 kapasitede havuzlanmış Spearman(B0) >= {S3_HAVUZ_ESIK}")

genel_pass = True
genel_ne   = False

if veri_eksik:
    yaz("\nKRİTİK HATA: Veri setinde eksiklik/tutarsızlık bulunduğu için hiçbir PASS/FAIL kararı üretilmemiştir.")
    yaz("S1, S2, S3 ve GENEL KARARIN tamamı NOT EVALUABLE olarak işaretlenmiştir.")
    genel_ne = True
    genel_pass = False

    for kap in KARAR_KAP:
        yaz(f"\n[{KARAR_SC}] Kapasite %{int(kap*100)}:")
        yaz("  S1: NOT EVALUABLE (Veri eksikliği/hatası nedeniyle değerlendirme yapılmadı)")
        yaz("  S2: NOT EVALUABLE (Veri eksikliği/hatası nedeniyle değerlendirme yapılmadı)")

    yaz(f"\n[{KARAR_SC}] S3 (bilgi amaçlı) Kapasite %90:")
    yaz("  S3: NOT EVALUABLE (Veri eksikliği/hatası nedeniyle değerlendirme yapılmadı)")
else:
    for kap in KAPASITE_ORANLARI:
        hav_row = kor_df[
            (kor_df["senaryo"] == KARAR_SC) &
            (kor_df["kapasite_orani"] == kap) &
            (kor_df["fold"] == "HAVUZ")
        ].iloc[0]

        kiv_rows = kor_df[
            (kor_df["senaryo"] == KARAR_SC) &
            (kor_df["kapasite_orani"] == kap) &
            (kor_df["fold"] != "HAVUZ")
        ]

        rho_b0_h   = float(hav_row["spearman_B0"])
        rho_prs_h  = float(hav_row["spearman_Pers"])
        rho_prf_h  = float(hav_row["spearman_Profil"])
        tanimli_b0_h = bool(hav_row["spearman_B0_tanimli"])
        tanimli_prs_h = bool(hav_row["spearman_Pers_tanimli"])
        tanimli_prf_h = bool(hav_row["spearman_Profil_tanimli"])

        kivrimda_tanimsiz = any(not bool(r["spearman_B0_tanimli"]) for _, r in kiv_rows.iterrows())

        yaz(f"\n[{KARAR_SC}] Kapasite %{int(kap*100)} = {hav_row['n_gun']:.0f} gün")
        yaz(f"  Havuz Spearman: B0={rho_b0_h:.3f} (tanımlı={tanimli_b0_h}) | "
            f"Pers={rho_prs_h:.3f} (tanımlı={tanimli_prs_h}) | "
            f"Profil={rho_prf_h:.3f} (tanımlı={tanimli_prf_h})")

        # S1 Değerlendirmesi
        if (not tanimli_b0_h) or (not tanimli_prs_h) or (not tanimli_prf_h):
            s1_str = "NOT EVALUABLE (B0 veya baseline Spearman tanımsız)"
            s1_ok  = False
            if kap in KARAR_KAP:
                genel_ne = True
        else:
            max_base = max(rho_prs_h, rho_prf_h)
            s1_ok    = rho_b0_h > max_base
            s1_str   = "PASS" if s1_ok else "FAIL"
            yaz(f"  S1: {s1_str} [B0={rho_b0_h:.3f} > max_baseline={max_base:.3f}]")

        if "NOT EVALUABLE" in s1_str:
            yaz(f"  S1: {s1_str}")

        # S2 Değerlendirmesi
        if (not tanimli_b0_h) or kivrimda_tanimsiz:
            s2_str = "NOT EVALUABLE (Havuzlanmış veya en az bir kıvrım Spearman tanımsız)"
            s2_ok  = False
            if kap in KARAR_KAP:
                genel_ne = True
        else:
            min_kiv = float(kiv_rows["spearman_B0"].min())
            s2a    = rho_b0_h >= S2_HAVUZ_ESIK
            s2b    = min_kiv  >= S2_KIVRIM_MIN
            s2_ok  = s2a and s2b
            s2_str = "PASS" if s2_ok else "FAIL"
            yaz(f"  S2: {s2_str} [havuz={rho_b0_h:.3f} >= {S2_HAVUZ_ESIK}: {'Evet' if s2a else 'HAYIR'} | "
                f"min_kıvrım={min_kiv:.3f} >= {S2_KIVRIM_MIN}: {'Evet' if s2b else 'HAYIR'}]")

        if "NOT EVALUABLE" in s2_str:
            yaz(f"  S2: {s2_str}")

        if kap in KARAR_KAP:
            if not (s1_ok and s2_ok):
                genel_pass = False

    # S3 Değerlendirmesi (Bilgi Amaçlı)
    kap_s3   = 0.90
    hav_s3   = kor_df[
        (kor_df["senaryo"] == KARAR_SC) &
        (kor_df["kapasite_orani"] == kap_s3) &
        (kor_df["fold"] == "HAVUZ")
    ].iloc[0]
    kiv_s3   = kor_df[
        (kor_df["senaryo"] == KARAR_SC) &
        (kor_df["kapasite_orani"] == kap_s3) &
        (kor_df["fold"] != "HAVUZ")
    ]

    kiv_s3_tanimsiz = any(not bool(r["spearman_B0_tanimli"]) for _, r in kiv_s3.iterrows())
    tanimli_s3_hav  = bool(hav_s3["spearman_B0_tanimli"])
    rho_s3          = float(hav_s3["spearman_B0"])

    yaz(f"\n[{KARAR_SC}] S3 (bilgi amaçlı) Kapasite %{int(kap_s3*100)}:")
    if (not tanimli_s3_hav) or kiv_s3_tanimsiz:
        yaz("  S3: NOT EVALUABLE (Havuzlanmış veya en az bir kıvrım Spearman tanımsız)")
    else:
        s3_ok = rho_s3 >= S3_HAVUZ_ESIK
        yaz(f"  S3: {'PASS' if s3_ok else 'FAIL'} [Spearman_B0={rho_s3:.3f} >= {S3_HAVUZ_ESIK}]")

# Genel Karar
yaz()
yaz("=" * 76)
if veri_eksik or genel_ne:
    yaz(f"GENEL KARAR ({KARAR_SC}, %70 ve %80 kapasite): NOT EVALUABLE")
    yaz("  (Veri eksikliği/tutarsızlığı veya en az bir kapasite seviyesinde tanımsız korelasyon.)")
else:
    yaz(f"GENEL KARAR ({KARAR_SC}, %70 ve %80 kapasite): {'PASS' if genel_pass else 'FAIL'}")
    if not genel_pass:
        yaz("  (S1 veya S2 en az bir kapasite düzeyinde sağlanamadı.)")
yaz("=" * 76)

# ============================================================
# BÖLÜM 4: DUYARLILIK SENARYOLARI (F50, F25) - BİLGİ AMAÇLI
# ============================================================
bolum("4. DUYARLILIK SENARYOLARI (F50, F25) - BİLGİ AMAÇLI")
yaz("Not: F50/F25 varsayımları mekanik avantaj üretir; yalnızca duyarlılık analizi amaçlıdır.")
if veri_eksik:
    yaz("Veri eksikliği/hatası nedeniyle duyarlılık korelasyonları hesaplanmadı.")
else:
    for sc in ["F50", "F25"]:
        yaz(f"\n[{sc}] Havuzlanmış Spearman:")
        hav = kor_df[(kor_df["senaryo"] == sc) & (kor_df["fold"] == "HAVUZ")][
            ["kapasite_orani", "n_gun", "n_sifir_gercek",
             "spearman_B0_tanimli", "spearman_B0",
             "spearman_Pers", "spearman_Profil", "pearson_B0"]
        ]
        yaz(hav.round(3).to_string(index=False))

# ============================================================
# BÖLÜM 5: ÇIKTI DOSYALARI
# ============================================================
bolum("5. ÇIKTI DOSYALARI")
yaz(f"  {OUT / 'siddet_gun_detay.csv'}     <- günlük aşım enerjisi detayları")
yaz(f"  {OUT / 'siddet_korelasyonlar.csv'} <- kıvrım ve havuzlanmış korelasyonlar")
yaz(f"  {OUT / 'siddet_ozeti.txt'}         <- bu özet rapor")

yol = OUT / "siddet_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nÖzet başarıyla kaydedildi: {yol}")
