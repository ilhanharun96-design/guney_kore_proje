# -*- coding: utf-8 -*-
"""
simulasyon_motoru.py — Aşama 3 Akıllı Şarj Simülasyon Motoru

Bu modül, akilli_sarj_simulasyonu.py dosyasındaki hesaplama mantığını
KORUYARAK, tek bir (senaryo, kapasite, esneklik, bekleme) kombinasyonu için
çağrılabilir fonksiyonlara dönüştürür.

Değişmez kurallar
-----------------
- akilli_sarj_simulasyonu.py DEĞİŞTİRİLMEZ; referans betik olarak kalır.
- Bu modül dosya yazmaz, log basmaz, ekrana çıktı vermez. Saf hesaplama.
- Eylül 2022 test verisi bu modülde KESİNLİKLE kullanılmaz.
  (K1: sim filtresi; K2: assert; K3: REF_BIT sınırı; K5/K6: ek tripwire)
- simule_et() ve kontrolsuz_yuk() mevcut koddan BİREBİR taşınmıştır.
- KONTROL 2 mevcut kodla aynı işlem sırasını izler.
- KONTROL 1, başarısız olursa RuntimeError fırlatır (kaynak betikte yalnızca
  uyarı basıyordu; bu, kütüphane fonksiyonu için bilinçli bir sıkılaştırmadır).

Cache tasarımı
--------------
- dosya_imzasi()         : cache'siz; her çağrıda dosya sistemini okur.
- veri_hazirla(imza)     : @st.cache_data; imza zorunlu anahtar.
- ref_kw_hesapla(sc, imza): @st.cache_data; imza zorunlu anahtar.
- calistir_simulasyon(...): @st.cache_data; imza zorunlu anahtar.

Cache'li fonksiyonların gövdelerinde stat() veya None kontrolü YOKTUR.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Yollar ve sabitler (akilli_sarj_simulasyonu.py ile birebir)
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
VERI = BASE / "data" / "ChargingRecords.csv"
YUK = BASE / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"

DT_DK = 15
DT = DT_DK / 60.0  # 0.25 saat
SIM_BAS = pd.Timestamp("2022-07-31 00:00")
DEG_BAS = pd.Timestamp("2022-08-01 00:00")
DEG_BIT = pd.Timestamp("2022-09-01 00:00")
REF_BAS = pd.Timestamp("2021-10-07 18:00")
REF_BIT = pd.Timestamp("2022-08-01 00:00")

SENARYOLAR = {
    "U100": (1.0, "yuk_U100_kwh"),
    "F50":  (0.5, "yuk_F50_kwh"),
    "F25":  (0.25, "yuk_F25_kwh"),
}

RAPOR_GECERLI = 72039

# P99 tripwire — app.py'deki P99_REFS ile bilinçli olarak kopyalanmıştır.
# İki kaynak farklılaşırsa motor erken hata verir; hesaplamaya girmez.
P99_BEKLENEN_KW = {
    "U100": 329.8,
    "F50":  386.2,
    "F25":  429.7,
}
TOLERANS_KW = 0.1


# -----------------------------------------------------------------------------
# Dosya imzası (cache'siz utility) ve CSV okuma
# -----------------------------------------------------------------------------
def dosya_imzasi():
    """
    Kaynak CSV dosyalarının (mtime, size) imzasını döndürür.
    Cache'lenmez; her çağrıda dosya sistemini okur.

    Dönüş: (veri_mtime, veri_size, yuk_mtime, yuk_size)

    Bu tuple, cache'li fonksiyonlara ZORUNLU parametre olarak geçirilir;
    böylece CSV dosyaları değiştiğinde cache anahtarı değişir ve yeni
    hesaplama yapılır. Yalnızca bir hesaplama isteği sırasında çağrılır;
    dosya değişikliği tek başına otomatik hesaplama başlatmaz.
    """
    vs = VERI.stat()
    ys = YUK.stat()
    return (vs.st_mtime, vs.st_size, ys.st_mtime, ys.st_size)


def oku(yol):
    """CSV'yi kodlama fallback'i ile okur (mevcut koddan birebir)."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(yol, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("CSV okunamadı (kodlama).")


# -----------------------------------------------------------------------------
# Simülasyon çekirdeği — mevcut koddan BİREBİR
# -----------------------------------------------------------------------------
def simule_et(a, nom_end, b, m, E, cap_step, T):
    """
    LLF (least laxity first) açgözlü denetleyici.
    Mevcut akilli_sarj_simulasyonu.py'den BİREBİR taşınmıştır.

    Parametreler adım cinsindendir. Dönüş:
        (rem, load, last, erteli)
        - load: adım başına ENERJİ (kWh), güç değil.
    """
    N = len(a)
    rem = E.copy()
    load = np.zeros(T)
    last = np.full(N, -1)
    erteli = np.zeros(N)
    for t in range(T):
        act = np.flatnonzero((a <= t) & (b > t) & (rem > 1e-9))
        if act.size == 0:
            continue
        lax = (b[act] - t) - rem[act] / m[act]
        o = np.argsort(lax, kind="stable")
        idx = act[o]
        d = np.minimum(m[idx], rem[idx])
        once = np.cumsum(d) - d
        x = np.minimum(d, np.maximum(cap_step - once, 0.0))
        rem[idx] -= x
        load[t] = x.sum()
        ver = x > 0
        if ver.any():
            gi = idx[ver]
            last[gi] = t
            gec = t >= nom_end[gi]
            erteli[gi[gec]] += x[ver][gec]
    return rem, load, last, erteli


def kontrolsuz_yuk(a, nom_end, m, T):
    """
    Kapasitesiz (kontrolsüz) yük serisi — mevcut koddan BİREBİR.
    Dönüş: adım başına ENERJİ (kWh); güç değil.
    """
    diff = np.zeros(T + 1)
    np.add.at(diff, a, m)
    np.add.at(diff, nom_end, -m)
    return np.cumsum(diff)[:T]


# -----------------------------------------------------------------------------
# Veri hazırlama — cache'li
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def veri_hazirla(dosya_imzasi):
    """
    Ham veriyi okur, temizler, filtreler ve motor için hazırlar.

    dosya_imzasi yalnızca cache anahtarıdır; gövdede KULLANILMAZ.
    Eylül 2022 test verisi K1/K2/K5/K6 ile dışlanır.
    """
    ham = oku(VERI)
    gerekli = ["StartDay", "StartTime", "EndDay", "EndTime", "Demand", "Duration"]
    eksik = [c for c in gerekli if c not in ham.columns]
    if eksik:
        raise RuntimeError(f"Eksik sütunlar: {eksik}")

    # --- Mevcut koddan birebir temizleme ---
    df = ham.drop_duplicates().copy()
    df["start"] = pd.to_datetime(
        df["StartDay"].astype(str).str.strip() + " " + df["StartTime"].astype(str).str.strip(),
        errors="coerce",
    )
    df["end"] = pd.to_datetime(
        df["EndDay"].astype(str).str.strip() + " " + df["EndTime"].astype(str).str.strip(),
        errors="coerce",
    )
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
    df = df.dropna(subset=["start", "end", "Demand", "Duration"]).copy()
    df["aralik_dk"] = (df["end"] - df["start"]).dt.total_seconds() / 60.0

    poz = df[(df["aralik_dk"] > 0) & (df["Duration"] > 0)]
    oran = float(np.median(poz["Duration"] / poz["aralik_dk"]))
    if not (0.9 < oran < 1.1):
        raise RuntimeError(f"Duration birimi dakika değil görünüyor (oran {oran:.3f}).")

    df["fark_dk"] = df["Duration"] - df["aralik_dk"]
    dis = (
        (df["Duration"] < 0)
        | (df["Duration"] == 0)
        | (df["fark_dk"] > 5)
        | (df["aralik_dk"] <= 0)
    )
    gecerli = df[~dis].copy()
    if len(gecerli) != RAPOR_GECERLI:
        raise RuntimeError(
            f"Geçerli oturum sayısı rapordan farklı: {len(gecerli)} != {RAPOR_GECERLI}"
        )

    # --- Simülasyon penceresi (K1) ---
    sim = gecerli[
        (gecerli["start"] >= SIM_BAS)
        & (gecerli["start"] < DEG_BIT)
        & (gecerli["Demand"] > 0)
    ].copy()
    sim = sim.sort_values("start").reset_index(drop=True)

    # --- Eylül 2022 güvenlik tripwire'ları ---
    if not (sim["start"].max() < DEG_BIT):                       # K2
        raise RuntimeError("Kritik: Eylül 2022 oturumu simülasyon verisine karışmış!")
    if DEG_BIT != pd.Timestamp("2022-09-01 00:00"):              # K5
        raise RuntimeError("DEG_BIT beklenen değerden farklı; test sınırı bozulmuş.")
    if not (REF_BIT <= DEG_BAS):                                  # K6
        raise RuntimeError("REF_BIT > DEG_BAS; referans penceresi değerlendirmeye taşıyor.")

    # --- Motor için türetilmiş diziler ---
    N = len(sim)
    a = (
        (sim["start"] - SIM_BAS).dt.total_seconds() // (DT_DK * 60)
    ).astype(int).to_numpy()
    n = np.maximum(1, np.round(sim["Duration"].to_numpy() / DT_DK)).astype(int)
    E = sim["Demand"].to_numpy(dtype=float)
    deg = (sim["start"] >= DEG_BAS).to_numpy()
    t_bas = int((DEG_BAS - SIM_BAS) / pd.Timedelta(minutes=DT_DK))
    t_bit = int((DEG_BIT - SIM_BAS) / pd.Timedelta(minutes=DT_DK))

    yuk = pd.read_csv(
        YUK, encoding="utf-8-sig", parse_dates=["datetime"], index_col="datetime"
    )

    return {
        "sim": sim,
        "a": a,
        "n": n,
        "E": E,
        "deg": deg,
        "t_bas": t_bas,
        "t_bit": t_bit,
        "yuk": yuk,
        "N": N,
        "deg_oturum": int(deg.sum()),
        "deg_enerji_kwh": float(E[deg].sum()),
    }


# -----------------------------------------------------------------------------
# Referans P99 — cache'li + tripwire
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def ref_kw_hesapla(senaryo, dosya_imzasi):
    """
    Senaryo için eğitim dönemi P99 referans gücünü (kW, HASSAS) hesaplar.
    P99_BEKLENEN_KW ile tripwire kontrolü yapar; tolerans aşılırsa RuntimeError.
    """
    if senaryo not in SENARYOLAR:
        raise ValueError(f"Bilinmeyen senaryo: {senaryo}")

    v = veri_hazirla(dosya_imzasi)
    yuk = v["yuk"]
    kolon = SENARYOLAR[senaryo][1]
    ref_kw = float(
        yuk.loc[(yuk.index >= REF_BAS) & (yuk.index < REF_BIT), kolon].quantile(0.99)
    )

    beklenen = P99_BEKLENEN_KW[senaryo]
    fark = abs(ref_kw - beklenen)
    if fark > TOLERANS_KW:
        raise RuntimeError(
            f"P99 tutarsızlığı: senaryo={senaryo}, hesaplanan={ref_kw:.4f}, "
            f"beklenen={beklenen:.1f}, fark={fark:.4f} kW > {TOLERANS_KW} kW"
        )
    return ref_kw


# -----------------------------------------------------------------------------
# Ana simülasyon fonksiyonu — cache'li
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def calistir_simulasyon(
    senaryo,
    kapasite_orani,
    esnek_pay,
    bekleme_saat,
    dosya_imzasi,
    tekrar=5,
    seed_temel=1000,
):
    """
    Tek bir (senaryo, kapasite, esneklik, bekleme) kombinasyonu için
    akıllı şarj simülasyonunu çalıştırır.

    Parametreler
    ------------
    senaryo        : "U100" | "F50" | "F25"
    kapasite_orani : >0 float; kapasite_kw = ref_kw_p99_hassas * kapasite_orani
    esnek_pay      : [0,1] float; 0 → tek tekrar
    bekleme_saat   : [0,2] float; esnek oturumlara ek süre
    dosya_imzasi   : (veri_mtime, veri_size, yuk_mtime, yuk_size) — ZORUNLU
    tekrar         : >=1 int; esnek_pay > 0 ise tekrar sayısı
    seed_temel     : RNG tohum tabanı (varsayılan 1000, mevcut kodla aynı)

    Dönüş
    -----
    dict — bkz. v4 tasarımı Bölüm 8.
    """
    # --- Girdi doğrulamaları ---
    if senaryo not in SENARYOLAR:
        raise ValueError(f"Bilinmeyen senaryo: {senaryo}")
    if kapasite_orani <= 0:
        raise ValueError(f"kapasite_orani > 0 olmalı: {kapasite_orani}")
    if not (0.0 <= esnek_pay <= 1.0):
        raise ValueError(f"esnek_pay [0,1] aralığında olmalı: {esnek_pay}")
    if not (0.0 <= bekleme_saat <= 2.0):
        raise ValueError(
            f"bekleme_saat [0, 2] saat aralığında olmalı: {bekleme_saat}"
        )
    if tekrar < 1:
        raise ValueError(f"tekrar >= 1 olmalı: {tekrar}")

    # --- Veri ve referans P99 ---
    v = veri_hazirla(dosya_imzasi)
    ref_kw_p99 = ref_kw_hesapla(senaryo, dosya_imzasi)
    ref_kw_p99_yuvarlanmis = P99_BEKLENEN_KW[senaryo]
    ref_kw_p99_fark = abs(ref_kw_p99 - ref_kw_p99_yuvarlanmis)

    a = v["a"]
    n = v["n"]
    E = v["E"]
    deg = v["deg"]
    t_bas = v["t_bas"]
    t_bit = v["t_bit"]
    yuk = v["yuk"]
    N = v["N"]
    top_e = v["deg_enerji_kwh"]         # = E[deg].sum()
    deg_sayi = float(v["deg_oturum"])   # = deg.sum()

    # --- Senaryo parametreleri (mevcut kodla birebir) ---
    f, kolon = SENARYOLAR[senaryo]
    nom = np.maximum(1, np.round(f * n)).astype(int)
    m = E / nom
    nom_end = a + nom
    b_tek = a + n
    T = int(b_tek.max() + 2 + int(2.0 / DT))

    kapasite_kw = ref_kw_p99 * kapasite_orani
    cap_step_kwh = kapasite_kw * DT

    # --- Kontrolsüz baseline (kapasiteden bağımsız) ---
    unc_kwh_adim = kontrolsuz_yuk(a, nom_end, m, T)  # kWh (15dk adım)
    unc_kw_adim = unc_kwh_adim / DT                   # kW (15dk ortalama)
    seg_kw = unc_kw_adim[t_bas:t_bit]                 # kW
    toplam_enerji_kwh = float(seg_kw.sum() * DT)
    donem_ortalama_kw = float(seg_kw.mean())
    donem_tepe_kw_15dk = float(seg_kw.max())

    # Aşım — mevcut kod işlem sırasıyla (kW farkı × DT = kWh)
    asim_kw = np.maximum(seg_kw - kapasite_kw, 0.0)
    asim_sure_saat = float((seg_kw > kapasite_kw).sum() * DT)
    asim_enerji_kwh = float(asim_kw.sum() * DT)
    asim_enerji_pct = (
        asim_enerji_kwh / toplam_enerji_kwh * 100.0 if toplam_enerji_kwh > 0 else 0.0
    )

    saatlik_kw_kontrolsuz = seg_kw.reshape(-1, 4).mean(axis=1)

    # --- KONTROL 1: sonsuz kapasite = kontrolsüz (başarısızsa RuntimeError) ---
    _, load_inf, _, _ = simule_et(a, nom_end, b_tek, m, E, np.inf, T)
    if not np.allclose(load_inf / DT, unc_kw_adim, atol=1e-6):
        raise RuntimeError(
            "KONTROL 1 başarısız: sonsuz kapasiteli simülasyon çıktısı, "
            "kontrolsüz yük serisiyle eşleşmedi (atol=1e-6)."
        )
    kontrol1_gecildi = True

    # --- KONTROL 2: saatlik yük CSV ile Pearson (eski kod sırasıyla) ---
    sa = seg_kw.reshape(-1, 4).mean(axis=1)
    hedef = yuk.loc[
        (yuk.index >= DEG_BAS) & (yuk.index < DEG_BIT), kolon
    ].to_numpy()
    if len(hedef) == len(sa):
        kontrol2_korelasyon = float(np.corrcoef(sa, hedef)[0, 1])
        kontrol2_gecildi = bool(kontrol2_korelasyon > 0.98)
    else:
        kontrol2_korelasyon = float("nan")
        kontrol2_gecildi = False

    # --- Tekrar döngüsü ---
    tekrar_efektif = tekrar if esnek_pay > 0 else 1
    tekrar_detay = []
    saatlik_kw_kontrollu_list = []

    for rep in range(tekrar_efektif):
        if esnek_pay > 0:
            rng = np.random.default_rng(seed_temel + rep)
            esnek = rng.random(N) < esnek_pay
        else:
            esnek = np.zeros(N, dtype=bool)

        b = a + n + np.where(esnek, int(round(bekleme_saat / DT)), 0)
        rem, load, last, erteli = simule_et(a, nom_end, b, m, E, cap_step_kwh, T)

        kw = load / DT

        # Bilimsel bütünlük assert'leri
        if kw.max() > kapasite_kw + 1e-6:
            raise RuntimeError(
                f"Kapasite aşıldı (hata): tepe={kw.max():.6f} kW > "
                f"kapasite={kapasite_kw:.6f} kW"
            )
        if not np.isclose(load.sum(), (E - rem).sum(), rtol=1e-9, atol=1e-6):
            raise RuntimeError("Enerji korunmadı (hata).")

        gec_adim = np.where(last >= 0, np.maximum(0, last + 1 - nom_end), 0)
        gecikmis = deg & (gec_adim > 0)
        gdk = gec_adim[gecikmis] * DT_DK

        karsilanamayan_pct_rep = float(rem[deg].sum()) / top_e * 100.0
        ertelenen_pct_rep = float(erteli[deg].sum()) / top_e * 100.0
        geciken_oturum_pct_rep = float(gecikmis.sum()) / deg_sayi * 100.0
        ort_gecikme_dk_rep = float(gdk.mean()) if gdk.size else 0.0
        p95_gecikme_dk_rep = float(np.quantile(gdk, 0.95)) if gdk.size else 0.0
        tepe_kw_kontrollu_rep = float(kw[t_bas:t_bit].max())
        saatlik_kw_kontrollu_rep = kw[t_bas:t_bit].reshape(-1, 4).mean(axis=1)

        tekrar_detay.append(
            {
                "rep": rep,
                "seed": (seed_temel + rep) if esnek_pay > 0 else None,
                "karsilanamayan_pct": karsilanamayan_pct_rep,
                "ertelenen_pct": ertelenen_pct_rep,
                "geciken_oturum_pct": geciken_oturum_pct_rep,
                "ort_gecikme_dk": ort_gecikme_dk_rep,
                "p95_gecikme_dk": p95_gecikme_dk_rep,
                "tepe_kw_kontrollu": tepe_kw_kontrollu_rep,
            }
        )
        saatlik_kw_kontrollu_list.append(saatlik_kw_kontrollu_rep)

    # --- Ortalamalar ve std'ler (numpy; ddof=1, pandas groupby().std() ile uyumlu) ---
    def _mean_std(key):
        arr = np.asarray([r[key] for r in tekrar_detay], dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return mu, sd

    karsilanamayan_mu, karsilanamayan_sd = _mean_std("karsilanamayan_pct")
    ertelenen_mu, ertelenen_sd = _mean_std("ertelenen_pct")
    geciken_mu, geciken_sd = _mean_std("geciken_oturum_pct")
    ort_gecikme_mu, ort_gecikme_sd = _mean_std("ort_gecikme_dk")
    p95_gecikme_mu, p95_gecikme_sd = _mean_std("p95_gecikme_dk")

    tepe_vals = np.asarray(
        [r["tepe_kw_kontrollu"] for r in tekrar_detay], dtype=float
    )
    tepe_kw_ortalama = float(tepe_vals.mean())
    tepe_kw_std = float(tepe_vals.std(ddof=1)) if tepe_vals.size > 1 else 0.0
    tepe_kw_maksimum = float(tepe_vals.max())
    tepe_kw_minimum = float(tepe_vals.min())

    saatlik_kw_ortalama = np.mean(np.vstack(saatlik_kw_kontrollu_list), axis=0)

    # --- Dönüş ---
    return {
        "meta": {
            "senaryo": senaryo,
            "ref_kw_p99": ref_kw_p99,
            "ref_kw_p99_yuvarlanmis": ref_kw_p99_yuvarlanmis,
            "ref_kw_p99_fark": ref_kw_p99_fark,
            "kapasite_orani_p99": float(kapasite_orani),
            "kapasite_kw": float(kapasite_kw),
            "esnek_pay": float(esnek_pay),
            "bekleme_saat": float(bekleme_saat),
            "tekrar_efektif": int(tekrar_efektif),
            "seed_temel": int(seed_temel),
            "dosya_imzasi": tuple(dosya_imzasi),
            "degerlendirme_bas": "2022-08-01",
            "degerlendirme_bit": "2022-09-01",
            "oturum_sayisi": int(v["deg_oturum"]),
            "toplam_enerji_kwh": float(top_e),
            "donem_ortalama_kw": donem_ortalama_kw,
            "donem_tepe_kw_15dk": donem_tepe_kw_15dk,
            "zaman_adimi_dk": DT_DK,
            "dt_saat": DT,
        },
        "kontrolsuz": {
            "asim_sure_saat": asim_sure_saat,
            "asim_enerji_kwh": asim_enerji_kwh,
            "asim_enerji_pct": asim_enerji_pct,
            "saatlik_kw": saatlik_kw_kontrolsuz,
        },
        "akilli": {
            "karsilanamayan_pct_ortalama": karsilanamayan_mu,
            "karsilanamayan_pct_std": karsilanamayan_sd,
            "ertelenen_pct_ortalama": ertelenen_mu,
            "ertelenen_pct_std": ertelenen_sd,
            "geciken_oturum_pct_ortalama": geciken_mu,
            "geciken_oturum_pct_std": geciken_sd,
            "ort_gecikme_dk_ortalama": ort_gecikme_mu,
            "ort_gecikme_dk_std": ort_gecikme_sd,
            "p95_gecikme_dk_ortalama": p95_gecikme_mu,
            "p95_gecikme_dk_std": p95_gecikme_sd,
            "tepe_kw_ortalama": tepe_kw_ortalama,
            "tepe_kw_std": tepe_kw_std,
            "tepe_kw_maksimum": tepe_kw_maksimum,
            "tepe_kw_minimum": tepe_kw_minimum,
            "saatlik_kw_ortalama": saatlik_kw_ortalama,
            "tekrar_detay": tekrar_detay,
        },
        "kontroller": {
            "kontrol1_gecildi": kontrol1_gecildi,
            "kontrol2_korelasyon": kontrol2_korelasyon,
            "kontrol2_gecildi": kontrol2_gecildi,
        },
    }