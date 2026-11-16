from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
OUT = BASE / "outputs_modeling" / "referans_ve_artik_denemeleri"
OUT.mkdir(parents=True, exist_ok=True)

HEDEFLER = ["session_start_count", "started_session_energy_kwh"]
FOLDLAR = [("2022-06-01", "2022-07-01"),
           ("2022-07-01", "2022-08-01"),
           ("2022-08-01", "2022-09-01")]

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def ref_model():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=42)


def artik_model():
    # Sabit, onceden belirlenmis ve bilincli olarak muhafazakar ayar (arama yapilmadi)
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=42)


def metrik(a, p, h24):
    hata = a - p
    a24, p24 = a[h24], p[h24]
    ga = a24 >= np.quantile(a24, 0.90)
    gp = p24 >= np.quantile(p24, 0.90)
    return {
        "MAE": float(np.mean(np.abs(hata))),
        "RMSE": float(np.sqrt(np.mean(hata ** 2))),
        "WAPE": float(np.sum(np.abs(hata)) / np.sum(a) * 100),
        "bias": float(np.mean(hata)),
        "MAE_h24": float(np.mean(np.abs(a24 - p24))),
        "zirve_recall_h24": float((ga & gp).sum() / max(int(ga.sum()), 1)),
    }


def baseline(tr, ev, hedef, hafta=None, w24=0.0):
    """Profil x Seviye. Profil yalnizca 'tr' icinden hesaplanir."""
    tr_u = tr.drop_duplicates("target_datetime")
    if hafta is not None:
        sinir = tr_u.target_datetime.max() - pd.Timedelta(weeks=hafta)
        tr_p = tr_u[tr_u.target_datetime > sinir]
    else:
        tr_p = tr_u
    prof = (tr_p.groupby(["target_weekday", "target_hour"])[hedef]
            .mean().rename("profil").reset_index())
    genel = tr_p[hedef].mean()
    pv = (ev[["target_weekday", "target_hour"]]
          .merge(prof, on=["target_weekday", "target_hour"], how="left")["profil"].to_numpy())
    lv168 = ev[f"{hedef}_rolling_mean_168"].to_numpy()
    lv24 = ev[f"{hedef}_rolling_mean_24"].to_numpy()
    seviye = (1 - w24) * lv168 + w24 * lv24
    return pv / genel * seviye


def artik_ozellikleri(df, hedef):
    lv = np.maximum(df[f"{hedef}_rolling_mean_168"].to_numpy(), 0.1)
    X = pd.DataFrame({
        "target_hour": df["target_hour"].to_numpy(),
        "target_weekday": df["target_weekday"].to_numpy(),
        "target_is_weekend": df["target_is_weekend"].to_numpy(),
        "forecast_horizon": df["forecast_horizon"].to_numpy(),
    })
    for k in [0, 1, 2, 3, 24, 48, 168]:
        X[f"lag{k}_rel"] = df[f"{hedef}_lag_{k}"].to_numpy() / lv
    X["rm24_rel"] = df[f"{hedef}_rolling_mean_24"].to_numpy() / lv
    X["std24_rel"] = df[f"{hedef}_rolling_std_24"].to_numpy() / lv
    X["std168_rel"] = df[f"{hedef}_rolling_std_168"].to_numpy() / lv
    return X


def poisson_taban(lam):
    """lam ortalama oldugunda Poisson icin beklenen mutlak sapma (medyan etrafinda) ortalamasi."""
    lam = np.clip(np.asarray(lam, dtype=float), 1e-6, None)
    kmax = int(max(60, np.max(lam) * 2 + 30))
    k = np.arange(0, kmax + 1)
    pmf = poisson.pmf(k[None, :], lam[:, None])
    med = poisson.ppf(0.5, lam)
    return float(np.mean((pmf * np.abs(k[None, :] - med[:, None])).sum(axis=1)))


# ---------------------------------------------------------------- veri
features = pd.read_csv(FE / "ozellik_listesi.csv")["feature_name"].tolist()
train = pd.read_csv(FE / "train_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
tum = pd.concat([train, val], ignore_index=True)
assert tum.target_datetime.max() < pd.Timestamp("2022-09-01"), "Test verisi karismis olabilir!"

yaz("Referans ve artik denemeleri (TEST VERISI KULLANILMAMISTIR)")
yaz(f"Kivrimlar (hedef ay): {[f[0][:7] for f in FOLDLAR]}")

sonuclar = []
taban_kayit = []
aralik_kayit = []

for hedef in HEDEFLER:
    sev = f"{hedef}_rolling_mean_168"
    for bas, bit in FOLDLAR:
        t0 = time.time()
        bas_t, bit_t = pd.Timestamp(bas), pd.Timestamp(bit)
        tr = tum[tum.target_datetime < bas_t]
        ev = tum[(tum.target_datetime >= bas_t) & (tum.target_datetime < bit_t)]
        y_tr = tr[hedef].to_numpy()
        y_ev = ev[hedef].to_numpy()
        h24 = (ev["forecast_horizon"].to_numpy() == 24)
        fold_ad = bas_t.strftime("%Y-%m")

        tahmin = {}
        b0_ev = baseline(tr, ev, hedef)
        tahmin["B0_profil_x_seviye"] = b0_ev
        tahmin["B1_seviye_168_24_karisim"] = baseline(tr, ev, hedef, w24=0.5)
        tahmin["B2_son8hafta_profil"] = baseline(tr, ev, hedef, hafta=8)
        tahmin["B3_son8hafta_ve_karisim"] = baseline(tr, ev, hedef, hafta=8, w24=0.5)

        m = ref_model().fit(tr[features], y_tr)
        tahmin["REF_45_ozellik"] = m.predict(ev[features])

        b0_tr = baseline(tr, tr, hedef)
        lv_tr = np.maximum(tr[sev].to_numpy(), 0.1)
        lv_ev = np.maximum(ev[sev].to_numpy(), 0.1)
        res_tr = (y_tr - b0_tr) / lv_tr
        m = artik_model().fit(artik_ozellikleri(tr, hedef), res_tr, sample_weight=lv_tr ** 2)
        tahmin["R1_B0_arti_artik_HGB"] = b0_ev + lv_ev * m.predict(artik_ozellikleri(ev, hedef))

        for ad, p in tahmin.items():
            p = np.clip(np.asarray(p, dtype=float), 0, None)
            satir = {"hedef": hedef, "fold": fold_ad, "yontem": ad,
                     "egitim_satiri": len(tr), "degerlendirme_satiri": len(ev)}
            satir.update(metrik(y_ev, p, h24))
            sonuclar.append(satir)

        if hedef == "session_start_count":
            taban_kayit.append({
                "fold": fold_ad,
                "poisson_taban_MAE": poisson_taban(np.clip(b0_ev, 0.01, None)),
                "B0_MAE": float(np.mean(np.abs(y_ev - np.clip(b0_ev, 0, None)))),
            })

        aralik_kayit.append({
            "hedef": hedef, "fold": fold_ad,
            "egitim_seviye_max": float(tr[sev].max()),
            "degerlendirme_seviye_ort": float(ev[sev].mean()),
            "degerlendirme_seviye_max": float(ev[sev].max()),
            "egitim_max_ustu_oran": float((ev[sev] > tr[sev].max()).mean()),
        })
        yaz(f"  tamamlandi: {hedef} / {fold_ad}  ({time.time() - t0:.0f} sn)")

df = pd.DataFrame(sonuclar)
df.to_csv(OUT / "sonuclar.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- ozet
KARSILASTIRILAN = ["B1_seviye_168_24_karisim", "B2_son8hafta_profil",
                   "B3_son8hafta_ve_karisim", "R1_B0_arti_artik_HGB"]
for hedef in HEDEFLER:
    d = df[df.hedef == hedef]
    yaz()
    yaz("=" * 72)
    yaz(f"HEDEF: {hedef}")
    yaz("=" * 72)
    for metr in ["MAE", "WAPE", "bias", "zirve_recall_h24"]:
        pv = d.pivot(index="yontem", columns="fold", values=metr)
        pv["ortalama"] = pv.mean(axis=1)
        pv = pv.sort_values("ortalama") if metr in ("MAE", "WAPE") else pv
        yaz(f"\n[{metr}]")
        yaz(pv.round(4).to_string())

    ref = d[d.yontem == "B0_profil_x_seviye"].set_index("fold")["MAE"]
    yaz("\nKarar kriteri (B0'a gore: ortalama MAE dusuk VE en az 2/3 kivrimda B0'dan iyi):")
    for yontem in KARSILASTIRILAN:
        mv = d[d.yontem == yontem].set_index("fold")["MAE"]
        iyi = int((mv < ref).sum())
        ok = mv.mean() < ref.mean()
        yaz(f"  {yontem}: ortalama MAE {mv.mean():.4f} vs B0 {ref.mean():.4f}; "
            f"{iyi}/3 kivrimda daha iyi -> {'KRITER SAGLANDI' if (ok and iyi >= 2) else 'saglanmadi'}")

yaz()
yaz("=" * 72)
yaz("POISSON TABANI (yalnizca oturum sayisi; YAKLASIK: B0 tahmini ortalama kabul edilir)")
yaz("=" * 72)
yaz(pd.DataFrame(taban_kayit).round(4).to_string(index=False))
yaz("Not: Veri asiri dagilimli olabilir ve B0 kendisi gurultuludur; bu deger kaba bir referanstir.")

yaz()
yaz("=" * 72)
yaz("SEVIYE ARALIGI KONTROLU (rolling_mean_168: egitim maksimumu ile karsilastirma)")
yaz("=" * 72)
yaz(pd.DataFrame(aralik_kayit).round(3).to_string(index=False))

yol = OUT / "referans_ve_artik_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")