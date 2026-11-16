from pathlib import Path
import time
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
VERI = BASE / "data" / "ChargingRecords.csv"
YUK = BASE / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"
OUT = BASE / "outputs_smart_charging"
OUT.mkdir(parents=True, exist_ok=True)

DT_DK = 15
DT = DT_DK / 60.0
SIM_BAS = pd.Timestamp("2022-07-31 00:00")
DEG_BAS = pd.Timestamp("2022-08-01 00:00")
DEG_BIT = pd.Timestamp("2022-09-01 00:00")
REF_BAS = pd.Timestamp("2021-10-07 18:00")
REF_BIT = pd.Timestamp("2022-08-01 00:00")

SENARYOLAR = {"U100": (1.0, "yuk_U100_kwh"),
              "F50": (0.5, "yuk_F50_kwh"),
              "F25": (0.25, "yuk_F25_kwh")}
KAPASITE_ORANLARI = [0.6, 0.7, 0.8, 0.9]
ESNEKLIK = [(0.0, 0.0), (0.25, 1.0), (0.25, 2.0), (0.5, 1.0), (0.5, 2.0)]  # (esnek pay, ek bekleme saat)
TEKRAR = 5
RAPOR_GECERLI = 72039

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def bolum(b):
    yaz()
    yaz("=" * 76)
    yaz(b)
    yaz("=" * 76)


def oku(yol):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(yol, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("CSV okunamadi (kodlama).")


def simule_et(a, nom_end, b, m, E, cap_step, T):
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
    diff = np.zeros(T + 1)
    np.add.at(diff, a, m)
    np.add.at(diff, nom_end, -m)
    return np.cumsum(diff)[:T]


# ------------------------------------------------------------ veri
bolum("1. VERI VE OTURUM KUMESI (orijinal dosya degistirilmez; TEST DONEMI KULLANILMAZ)")
ham = oku(VERI)
gerekli = ["StartDay", "StartTime", "EndDay", "EndTime", "Demand", "Duration"]
eksik = [c for c in gerekli if c not in ham.columns]
if eksik:
    raise SystemExit(f"Eksik sutunlar: {eksik}")
df = ham.drop_duplicates().copy()
df["start"] = pd.to_datetime(df["StartDay"].astype(str).str.strip() + " " + df["StartTime"].astype(str).str.strip(),
                             errors="coerce")
df["end"] = pd.to_datetime(df["EndDay"].astype(str).str.strip() + " " + df["EndTime"].astype(str).str.strip(),
                           errors="coerce")
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
df = df.dropna(subset=["start", "end", "Demand", "Duration"]).copy()
df["aralik_dk"] = (df["end"] - df["start"]).dt.total_seconds() / 60.0
poz = df[(df["aralik_dk"] > 0) & (df["Duration"] > 0)]
oran = float(np.median(poz["Duration"] / poz["aralik_dk"]))
if not (0.9 < oran < 1.1):
    raise SystemExit(f"Duration birimi dakika degil gorunuyor (oran {oran:.3f}).")
df["fark_dk"] = df["Duration"] - df["aralik_dk"]
dis = (df["Duration"] < 0) | (df["Duration"] == 0) | (df["fark_dk"] > 5) | (df["aralik_dk"] <= 0)
gecerli = df[~dis].copy()
yaz(f"Gecerli oturum: {len(gecerli)} (rapor: {RAPOR_GECERLI})")
if len(gecerli) != RAPOR_GECERLI:
    raise SystemExit("Gecerli oturum sayisi rapordan farkli; once bu incelenmeli.")

sim = gecerli[(gecerli["start"] >= SIM_BAS) & (gecerli["start"] < DEG_BIT) & (gecerli["Demand"] > 0)].copy()
sim = sim.sort_values("start").reset_index(drop=True)
assert sim["start"].max() < DEG_BIT, "Test donemi oturumu karismis olabilir!"
N = len(sim)
a = ((sim["start"] - SIM_BAS).dt.total_seconds() // (DT_DK * 60)).astype(int).to_numpy()
n = np.maximum(1, np.round(sim["Duration"].to_numpy() / DT_DK)).astype(int)
E = sim["Demand"].to_numpy(dtype=float)
deg = (sim["start"] >= DEG_BAS).to_numpy()
t_bas = int((DEG_BAS - SIM_BAS) / pd.Timedelta(minutes=DT_DK))
t_bit = int((DEG_BIT - SIM_BAS) / pd.Timedelta(minutes=DT_DK))
yaz(f"Simule edilen oturum: {N} (degerlendirme oturumu: {int(deg.sum())}); "
    f"degerlendirme enerjisi: {E[deg].sum():.1f} kWh")
yaz(f"Zaman adimi: {DT_DK} dk; degerlendirme adimlari: {t_bas}..{t_bit} ({t_bit - t_bas} adim)")

yuk = pd.read_csv(YUK, encoding="utf-8-sig", parse_dates=["datetime"], index_col="datetime")

kayit = []
t0 = time.time()
for sc, (f, kolon) in SENARYOLAR.items():
    nom = np.maximum(1, np.round(f * n)).astype(int)
    m = E / nom
    nom_end = a + nom
    ref_kw = float(yuk.loc[(yuk.index >= REF_BAS) & (yuk.index < REF_BIT), kolon].quantile(0.99))

    b_tek = a + n
    T = int(b_tek.max() + 2 + int(2.0 / DT))
    unc_kw = kontrolsuz_yuk(a, nom_end, m, T) / DT
    seg = unc_kw[t_bas:t_bit]
    ort_kw = float(seg.mean())

    bolum(f"SENARYO {sc}  (f={f})  referans (egitim doneminin saatlik %99'u) = {ref_kw:.1f} kW; "
          f"degerlendirme donemi kontrolsuz ortalama = {ort_kw:.1f} kW, tepe (15 dk) = {seg.max():.1f} kW")

    # kontrol 1: sonsuz kapasite = kontrolsuz
    _, load_inf, _, _ = simule_et(a, nom_end, b_tek, m, E, np.inf, T)
    yaz(f"  KONTROL 1 (sonsuz kapasite = kontrolsuz yuk): "
        f"{'GECTI' if np.allclose(load_inf / DT, unc_kw, atol=1e-6) else 'KALDI'}")
    # kontrol 2: saatlik karsilastirma
    sa = seg.reshape(-1, 4).mean(axis=1)
    hedef = yuk.loc[(yuk.index >= DEG_BAS) & (yuk.index < DEG_BIT), kolon].to_numpy()
    if len(hedef) == len(sa):
        kor = float(np.corrcoef(sa, hedef)[0, 1])
        yaz(f"  KONTROL 2 (saatlik yuk csv ile): korelasyon={kor:.4f}, toplam enerji orani={sa.sum() / hedef.sum():.4f} "
            f"{'GECTI' if kor > 0.98 else 'UYARI: dusuk korelasyon'}")
    else:
        yaz(f"  KONTROL 2 atlandi (saat sayisi farkli: {len(sa)} vs {len(hedef)})")

    for kap in KAPASITE_ORANLARI:
        cap_kw = kap * ref_kw
        cap_step = cap_kw * DT
        asim = np.maximum(seg - cap_kw, 0.0)
        unc_saat = float((seg > cap_kw).sum() * DT)
        unc_enerji = float(asim.sum() * DT)
        unc_pct = unc_enerji / float(seg.sum() * DT) * 100

        for pay, bekle in ESNEKLIK:
            tekrarlar = range(TEKRAR) if pay > 0 else [0]
            for rep in tekrarlar:
                if pay > 0:
                    esnek = np.random.default_rng(1000 + rep).random(N) < pay
                else:
                    esnek = np.zeros(N, dtype=bool)
                b = a + n + np.where(esnek, int(round(bekle / DT)), 0)
                rem, load, last, erteli = simule_et(a, nom_end, b, m, E, cap_step, T)

                kw = load / DT
                assert kw.max() <= cap_kw + 1e-6, "Kapasite asildi (hata)"
                assert np.isclose(load.sum(), (E - rem).sum(), rtol=1e-9, atol=1e-6), "Enerji korunmadi"

                top_e = float(E[deg].sum())
                gec_adim = np.where(last >= 0, np.maximum(0, last + 1 - nom_end), 0)
                gecikmis = deg & (gec_adim > 0)
                gdk = gec_adim[gecikmis] * DT_DK
                kayit.append({
                    "senaryo": sc, "kapasite_orani": kap, "kapasite_kw": cap_kw,
                    "esnek_pay": pay, "bekleme_saat": bekle, "tekrar": rep,
                    "kontrolsuz_asim_saat": unc_saat,
                    "kontrolsuz_asim_enerji_pct": unc_pct,
                    "karsilanamayan_pct": float(rem[deg].sum()) / top_e * 100,
                    "ertelenen_pct": float(erteli[deg].sum()) / top_e * 100,
                    "geciken_oturum_pct": float(gecikmis.sum()) / float(deg.sum()) * 100,
                    "ort_gecikme_dk": float(gdk.mean()) if gdk.size else 0.0,
                    "p95_gecikme_dk": float(np.quantile(gdk, 0.95)) if gdk.size else 0.0,
                    "tepe_kw_kontrollu": float(kw[t_bas:t_bit].max()),
                })
        yaz(f"  kapasite %{int(kap * 100)} ({cap_kw:.0f} kW, donem ortalamasinin {cap_kw / ort_kw:.2f} kati) tamamlandi "
            f"[{time.time() - t0:.0f} sn]")

res = pd.DataFrame(kayit)
res.to_csv(OUT / "akilli_sarj_ham_sonuclar.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------ ozet tablolar
sayisal = ["kapasite_kw", "kontrolsuz_asim_saat", "kontrolsuz_asim_enerji_pct", "karsilanamayan_pct",
           "ertelenen_pct", "geciken_oturum_pct", "ort_gecikme_dk", "p95_gecikme_dk"]
anahtar = ["senaryo", "kapasite_orani", "esnek_pay", "bekleme_saat"]
ozet = res.groupby(anahtar)[sayisal].mean()
ozet["karsilanamayan_std"] = res.groupby(anahtar)["karsilanamayan_pct"].std().fillna(0.0)
ozet = ozet.reset_index()
ozet.to_csv(OUT / "akilli_sarj_ozet_tablo.csv", index=False, encoding="utf-8-sig")

for sc in SENARYOLAR:
    bolum(f"OZET TABLO - {sc} (Agustos 2022; esnek olmayan satirlar tek kosu, digerleri {TEKRAR} tekrar ortalamasi)")
    t = ozet[ozet.senaryo == sc].copy()
    t["kap_%"] = (t["kapasite_orani"] * 100).round(0).astype(int)
    t = t[["kap_%", "kapasite_kw", "esnek_pay", "bekleme_saat", "kontrolsuz_asim_saat",
           "kontrolsuz_asim_enerji_pct", "karsilanamayan_pct", "karsilanamayan_std",
           "ertelenen_pct", "geciken_oturum_pct", "ort_gecikme_dk", "p95_gecikme_dk"]]
    yaz(t.round(2).to_string(index=False))

yaz()
yaz("Aciklama: kontrolsuz_asim_* = kapasite olmasaydi kapasiteyi asacak saat/enerji (kontrolsuz sarj).")
yaz("karsilanamayan_pct = son teslim zamaninda tamamlanamayan enerji / toplam enerji.")
yaz("ertelenen_pct = oturumun nominal sarj penceresinden SONRA teslim edilen enerji / toplam enerji.")
yaz("Hepsi senaryo varsayimlarina (f, esnek pay, ek bekleme) ve tek bir ag-toplam kapasitesine baglidir.")

yol = OUT / "akilli_sarj_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")