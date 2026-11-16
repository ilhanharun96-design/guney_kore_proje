from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
OUT = BASE / "outputs_modeling" / "iyilestirme_denemeleri"
OUT.mkdir(parents=True, exist_ok=True)

HEDEFLER = {
    "session_start_count": "session_start_count_rolling_mean_168",
    "started_session_energy_kwh": "started_session_energy_kwh_rolling_mean_168",
}
FOLDLAR = [("2022-06-01", "2022-07-01"),
           ("2022-07-01", "2022-08-01"),
           ("2022-08-01", "2022-09-01")]
RAPOR_AGUSTOS_MAE = {"session_start_count": 3.3427, "started_session_energy_kwh": 66.1994}

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def yeni_model():
    return HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=False, random_state=42,
    )


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


# ---------------------------------------------------------------- veri
features = pd.read_csv(FE / "ozellik_listesi.csv")["feature_name"].tolist()
train = pd.read_csv(FE / "train_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
tum = pd.concat([train, val], ignore_index=True)
assert tum.target_datetime.max() < pd.Timestamp("2022-09-01"), "Test verisi karismis olabilir!"

proxy = [f for f in features if ("day_of_year" in f) or ("month" in f)]
ozellik_E = [f for f in features if f not in proxy]

yaz("Rolling-origin ablasyon (TEST VERISI KULLANILMAMISTIR)")
yaz(f"Toplam ozellik: {len(features)}  Cikarilan takvim-indeks ozellikleri ({len(proxy)}): {proxy}")
yaz(f"E1/E2 ozellik sayisi: {len(ozellik_E)}")
yaz(f"Kivrimlar (hedef ay): {[f[0][:7] for f in FOLDLAR]}")

sonuclar = []
for hedef, seviye in HEDEFLER.items():
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

        m = yeni_model().fit(tr[features], y_tr)
        tahmin["REF_45_ozellik"] = m.predict(ev[features])

        m = yeni_model().fit(tr[ozellik_E], y_tr)
        tahmin["E1_takvim_indeksi_cikarildi"] = m.predict(ev[ozellik_E])

        lv_tr = np.maximum(tr[seviye].to_numpy(), 0.1)
        lv_ev = np.maximum(ev[seviye].to_numpy(), 0.1)
        m = yeni_model().fit(tr[ozellik_E], y_tr / lv_tr, sample_weight=lv_tr ** 2)
        tahmin["E2_oran_hedef_agirlikli"] = m.predict(ev[ozellik_E]) * lv_ev

        tr_u = tr.drop_duplicates("target_datetime")
        prof = (tr_u.groupby(["target_weekday", "target_hour"])[hedef]
                .mean().rename("profil").reset_index())
        genel = tr_u[hedef].mean()
        pv = (ev[["target_weekday", "target_hour"]]
              .merge(prof, on=["target_weekday", "target_hour"], how="left")["profil"].to_numpy())
        tahmin["BASE_profil_statik"] = pv
        tahmin["BASE_profil_x_seviye"] = pv / genel * ev[seviye].to_numpy()

        for ad, p in tahmin.items():
            p = np.clip(np.asarray(p, dtype=float), 0, None)
            satir = {"hedef": hedef, "fold": fold_ad, "yontem": ad,
                     "egitim_satiri": len(tr), "degerlendirme_satiri": len(ev)}
            satir.update(metrik(y_ev, p, h24))
            sonuclar.append(satir)

        yaz(f"  tamamlandi: {hedef} / {fold_ad}  ({time.time() - t0:.0f} sn)")

df = pd.DataFrame(sonuclar)
df.to_csv(OUT / "rolling_ablasyon_sonuclari.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- ozet
for hedef in HEDEFLER:
    d = df[df.hedef == hedef]
    yaz()
    yaz("=" * 72)
    yaz(f"HEDEF: {hedef}")
    yaz("=" * 72)

    ref_ag = d[(d.yontem == "REF_45_ozellik") & (d.fold == "2022-08")]["MAE"].iloc[0]
    yaz(f"Kontrol: REF Agustos MAE = {ref_ag:.4f}  | rapordaki dogrulama MAE = {RAPOR_AGUSTOS_MAE[hedef]}")

    for metr in ["MAE", "WAPE", "bias", "zirve_recall_h24"]:
        pv = d.pivot(index="yontem", columns="fold", values=metr)
        pv["ortalama"] = pv.mean(axis=1)
        pv = pv.sort_values("ortalama") if metr in ("MAE", "WAPE") else pv
        yaz(f"\n[{metr}]")
        yaz(pv.round(4).to_string())

    ref = d[d.yontem == "REF_45_ozellik"].set_index("fold")["MAE"]
    yaz("\nKarar kriteri (ortalama MAE < REF ve en az 2/3 kivrimda REF'ten iyi):")
    for yontem in ["E1_takvim_indeksi_cikarildi", "E2_oran_hedef_agirlikli"]:
        mv = d[d.yontem == yontem].set_index("fold")["MAE"]
        iyi = int((mv < ref).sum())
        ort_ok = mv.mean() < ref.mean()
        yaz(f"  {yontem}: ortalama MAE {mv.mean():.4f} vs REF {ref.mean():.4f}; "
            f"{iyi}/3 kivrimda daha iyi -> {'KRITER SAGLANDI' if (ort_ok and iyi >= 2) else 'saglanmadi'}")

yol = OUT / "iyilestirme_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")