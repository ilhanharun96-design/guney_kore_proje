from pathlib import Path
import numpy as np
import pandas as pd
import joblib

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
MD = BASE / "outputs_modeling"
OUT = MD / "dogrulama_tani"
OUT.mkdir(parents=True, exist_ok=True)

HEDEFLER = {
    "session_start_count": {
        "rapor_val_mae": 3.3427,
        "seviye": "session_start_count_rolling_mean_168",
    },
    "started_session_energy_kwh": {
        "rapor_val_mae": 66.1994,
        "seviye": "started_session_energy_kwh_rolling_mean_168",
    },
}

SATIRLAR = []


def yaz(metin=""):
    print(metin)
    SATIRLAR.append(str(metin))


def bolum(baslik):
    yaz()
    yaz("=" * 72)
    yaz(baslik)
    yaz("=" * 72)


def mae(a, p):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(p))))


def wape(a, p):
    a = np.asarray(a)
    p = np.asarray(p)
    return float(np.sum(np.abs(a - p)) / np.sum(np.abs(a)) * 100)


def model_yukle(hedef):
    yol = MD / "models" / f"{hedef}_HistGradientBoosting.joblib"
    obj = joblib.load(yol)
    if hasattr(obj, "predict"):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            if hasattr(v, "predict"):
                return v
    raise RuntimeError(f"Model nesnesi bulunamadi: {yol} (tip: {type(obj)})")


# ---------------------------------------------------------------- veri
features = pd.read_csv(FE / "ozellik_listesi.csv")["feature_name"].tolist()
train = pd.read_csv(FE / "train_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])

yaz("Dogrulama tani betigi (TEST VERISI KULLANILMAMISTIR, MODEL YENIDEN EGITILMEMISTIR)")
yaz(f"Egitim satiri: {len(train)}  Dogrulama satiri: {len(val)}  Ozellik sayisi: {len(features)}")
yaz(f"Dogrulama hedef araligi: {val.target_datetime.min()} - {val.target_datetime.max()}")

D = {}
for hedef, ayar in HEDEFLER.items():
    model = model_yukle(hedef)
    cols = list(getattr(model, "feature_names_in_", features))
    ham = model.predict(val[cols])
    pred = np.clip(ham, 0, None)
    df = pd.DataFrame({
        "target_datetime": val["target_datetime"].to_numpy(),
        "forecast_horizon": val["forecast_horizon"].to_numpy(),
        "target_hour": val["target_hour"].to_numpy(),
        "actual": val[hedef].to_numpy(),
        "pred": pred,
    })
    D[hedef] = {"model": model, "cols": cols, "df": df, "ham": ham}

# ------------------------------------------------ 1. yeniden uretim kontrolu
bolum("1. KAYITLI MODEL KONTROLU (rapordaki dogrulama MAE ile karsilastirma)")
for hedef, ayar in HEDEFLER.items():
    d = D[hedef]["df"]
    ham = D[hedef]["ham"]
    yaz(f"[{hedef}]")
    yaz(f"  Ham negatif tahmin sayisi: {int((ham < 0).sum())}")
    yaz(f"  MAE (0'a kirpilmis): {mae(d.actual, d.pred):.4f}   MAE (ham): {mae(d.actual, ham):.4f}   Rapor: {ayar['rapor_val_mae']}")
    fark = abs(mae(d.actual, d.pred) - ayar["rapor_val_mae"])
    yaz("  Sonuc: " + ("UYUMLU" if fark < 0.01 else "UYUMSUZ -> modeli/ozellik siralamasini kontrol edin"))

# ------------------------------------------------ 2. ufuk tutarliligi
bolum("2. UFUKLAR ARASI TUTARLILIK (model gecikmeli degerleri ne kadar kullaniyor?)")
for hedef in HEDEFLER:
    d = D[hedef]["df"]
    yaz(f"[{hedef}]")
    for h in [1, 6, 12, 18, 24]:
        s = d[d.forecast_horizon == h]
        yaz(f"  h={h:2d}: MAE={mae(s.actual, s.pred):.4f}  WAPE=%{wape(s.actual, s.pred):.2f}  n={len(s)}")
    g = d.groupby("target_datetime")["pred"].agg(["min", "max", "count"])
    g = g[g["count"] == 24]
    yay = g["max"] - g["min"]
    yaz(f"  Ayni hedef saat icin 24 ufuk tahmini arasindaki (max-min): ortalama {yay.mean():.3f}, medyan {yay.median():.3f}")
    yaz(f"  Karsilastirma icin gercek ortalama: {d.actual.mean():.3f}  (hedef saat sayisi: {len(g)})")

# ------------------------------------------------ 3. saat bazli yanlilik
bolum("3. SAAT BAZLI YANLILIK (dogrulama, forecast_horizon=24; hata = gercek - tahmin)")
for hedef in HEDEFLER:
    d = D[hedef]["df"]
    s = d[d.forecast_horizon == 24].copy()
    s["hata"] = s.actual - s.pred
    t = s.groupby("target_hour").agg(
        n=("actual", "size"),
        gercek_ort=("actual", "mean"),
        tahmin_ort=("pred", "mean"),
        ort_hata=("hata", "mean"),
        MAE=("hata", lambda x: float(np.mean(np.abs(x)))),
    )
    yaz(f"[{hedef}]  genel ort_hata={s.hata.mean():.3f}  MAE={s.hata.abs().mean():.3f}")
    yaz(t.round(2).to_string())
    t.to_csv(OUT / f"saatlik_yanlilik_{hedef}.csv", encoding="utf-8-sig")

# ------------------------------------------------ 4. kalibrasyon ve zirve
bolum("4. KALIBRASYON (tahmin degerine gore 10 dilim, tum ufuklar) VE ZIRVE YAKALAMA")
for hedef in HEDEFLER:
    d = D[hedef]["df"].copy()
    d["dilim"] = pd.qcut(d.pred, 10, duplicates="drop")
    c = d.groupby("dilim", observed=True).agg(
        n=("actual", "size"), tahmin_ort=("pred", "mean"), gercek_ort=("actual", "mean")
    )
    c["fark_gercek_eksi_tahmin"] = c.gercek_ort - c.tahmin_ort
    yaz(f"[{hedef}]")
    yaz(c.round(2).to_string())
    c.reset_index().astype({"dilim": str}).to_csv(
        OUT / f"kalibrasyon_{hedef}.csv", index=False, encoding="utf-8-sig"
    )

    s = d[d.forecast_horizon == 24]
    ga = s.actual >= s.actual.quantile(0.90)
    gp = s.pred >= s.pred.quantile(0.90)
    kesisim = int((ga & gp).sum())
    yaz(f"  Zirve yakalama (h=24, ust %10): gercek zirve={int(ga.sum())}, tahmin zirvesi={int(gp.sum())}, "
        f"kesisim={kesisim}, precision={kesisim / max(int(gp.sum()), 1):.2f}, recall={kesisim / max(int(ga.sum()), 1):.2f}")

# ------------------------------------------------ 5. basit referanslar
bolum("5. BASIT REFERANS YONTEMLERLE KARSILASTIRMA (dogrulama, tum ufuklar)")
for hedef, ayar in HEDEFLER.items():
    seviye = ayar["seviye"]
    tr_u = train.drop_duplicates("target_datetime")
    prof = (
        tr_u.groupby(["target_weekday", "target_hour"])[hedef]
        .mean().rename("profil").reset_index()
    )
    genel = tr_u[hedef].mean()
    v = val[["target_weekday", "target_hour", seviye, hedef]].merge(
        prof, on=["target_weekday", "target_hour"], how="left"
    )
    a = v[hedef].to_numpy()
    p_stat = v["profil"].to_numpy()
    p_seviye = (v["profil"] / genel * v[seviye]).to_numpy()
    d = D[hedef]["df"]
    yaz(f"[{hedef}]")
    for ad, p in [
        ("HistGradientBoosting (kayitli)", d.pred.to_numpy()),
        ("Profil_Statik (egitim, hafta-saat ortalamasi)", p_stat),
        ("Profil_x_Seviye (profil x son 168 saat ortalamasi)", p_seviye),
    ]:
        yaz(f"  {ad:55s} MAE={mae(a, p):9.4f}  WAPE=%{wape(a, p):6.2f}  ort_hata(gercek-tahmin)={np.mean(a - p):8.3f}")

# ------------------------------------------------ 6. permutation importance
bolum("6. PERMUTATION IMPORTANCE (dogrulama, tum ufuklar, MAE artisi; 3 tekrar)")


def grup_adi(f):
    gun_ay = ("day_of_year" in f) or ("month" in f)
    if f.startswith("target_"):
        return "hedef_gun_ay" if gun_ay else "hedef_saat_hafta"
    if f.startswith("origin_"):
        return "kaynak_gun_ay" if gun_ay else "kaynak_saat_hafta"
    if f == "forecast_horizon":
        return "forecast_horizon"
    if f.startswith("session_start_count_lag"):
        return "oturum_lag"
    if f.startswith("session_start_count_rolling"):
        return "oturum_rolling"
    if f.startswith("started_session_energy_kwh_lag"):
        return "enerji_lag"
    if f.startswith("started_session_energy_kwh_rolling"):
        return "enerji_rolling"
    return "diger"


def perm_mae(model, X, y, kolonlar, rng, tekrar=3):
    n = len(X)
    sonuc = []
    for _ in range(tekrar):
        Xp = X.copy()
        idx = rng.permutation(n)
        for c in kolonlar:
            Xp[c] = X[c].to_numpy()[idx]
        sonuc.append(mae(y, np.clip(model.predict(Xp), 0, None)))
    return float(np.mean(sonuc))


for hedef in HEDEFLER:
    model = D[hedef]["model"]
    cols = D[hedef]["cols"]
    X = val[cols].copy()
    y = val[hedef].to_numpy()
    rng = np.random.default_rng(42)
    baz = mae(y, np.clip(model.predict(X), 0, None))

    gruplar = {}
    for f in cols:
        gruplar.setdefault(grup_adi(f), []).append(f)

    yaz(f"[{hedef}]  temel MAE={baz:.4f}")
    yaz("  Grup onemi (MAE artisi):")
    gs = []
    for g, fl in gruplar.items():
        gs.append((g, len(fl), perm_mae(model, X, y, fl, rng) - baz))
    for g, k, art in sorted(gs, key=lambda x: -x[2]):
        yaz(f"    {g:20s} ({k:2d} ozellik)  +{art:8.4f}")

    tekler = [(f, perm_mae(model, X, y, [f], rng) - baz) for f in cols]
    tekler.sort(key=lambda x: -x[1])
    yaz("  Tek ozellik onemi (ilk 15):")
    for f, art in tekler[:15]:
        yaz(f"    {f:50s} +{art:8.4f}")
    pd.DataFrame(tekler, columns=["ozellik", "mae_artisi"]).to_csv(
        OUT / f"permutation_{hedef}.csv", index=False, encoding="utf-8-sig"
    )

# ------------------------------------------------ 7. kapsam ve kayma
bolum("7. TAKVIM KAPSAMI VE AYLIK DUZEY (egitim ici tahminler egitim verisi uzerindedir)")
tr_u = train.drop_duplicates("target_datetime")
va_u = val.drop_duplicates("target_datetime")
yaz(f"Egitim hedef araligi: {tr_u.target_datetime.min()} - {tr_u.target_datetime.max()}  "
    f"(target_day_of_year {tr_u.target_day_of_year.min()}-{tr_u.target_day_of_year.max()})")
yaz(f"Dogrulama hedef araligi: {va_u.target_datetime.min()} - {va_u.target_datetime.max()}  "
    f"(target_day_of_year {va_u.target_day_of_year.min()}-{va_u.target_day_of_year.max()})")
tr_aylar = set(tr_u.target_month.unique())
va_aylar = set(va_u.target_month.unique())
yaz(f"Egitimde bulunan hedef ay numaralari: {sorted(tr_aylar)}")
yaz(f"Dogrulamada olup egitimde hic olmayan ay numaralari: {sorted(va_aylar - tr_aylar)}")

for hedef in HEDEFLER:
    model = D[hedef]["model"]
    cols = D[hedef]["cols"]
    tr24 = train[train.forecast_horizon == 24]
    p_tr = np.clip(model.predict(tr24[cols]), 0, None)
    a = pd.DataFrame({
        "ay": tr24.target_datetime.dt.to_period("M").astype(str).to_numpy(),
        "gercek": tr24[hedef].to_numpy(),
        "tahmin": p_tr,
        "donem": "egitim",
    })
    dv = D[hedef]["df"]
    dv = dv[dv.forecast_horizon == 24]
    b = pd.DataFrame({
        "ay": dv.target_datetime.dt.to_period("M").astype(str).to_numpy(),
        "gercek": dv.actual.to_numpy(),
        "tahmin": dv.pred.to_numpy(),
        "donem": "dogrulama",
    })
    m = pd.concat([a, b]).groupby(["ay", "donem"]).agg(
        n=("gercek", "size"), gercek_ort=("gercek", "mean"), tahmin_ort=("tahmin", "mean")
    )
    m["fark_gercek_eksi_tahmin"] = m.gercek_ort - m.tahmin_ort
    yaz(f"[{hedef}] (h=24 satirlari)")
    yaz(m.round(2).to_string())
    m.to_csv(OUT / f"aylik_duzey_{hedef}.csv", encoding="utf-8-sig")

yol = OUT / "dogrulama_tani_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")