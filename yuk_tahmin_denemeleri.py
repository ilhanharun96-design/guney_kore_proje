from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
LP = BASE / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"
OUT = BASE / "outputs_load_profile" / "yuk_tahmin_denemeleri"
OUT.mkdir(parents=True, exist_ok=True)

TEST_BASLANGIC = pd.Timestamp("2022-09-01")
PROFIL_BASLANGIC = pd.Timestamp("2021-10-07 18:00")
SERILER = {"U100": "yuk_U100_kwh", "F50": "yuk_F50_kwh", "F25": "yuk_F25_kwh"}
FOLDLAR = [("2022-06-01", "2022-07-01"),
           ("2022-07-01", "2022-08-01"),
           ("2022-08-01", "2022-09-01")]
BOSLUK = pd.Timedelta(days=2)
KAL_HAFTA = 8
QS = [0.80, 0.90, 0.95]
SEVIYE = "started_session_energy_kwh_rolling_mean_168"

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def bolum(b):
    yaz()
    yaz("=" * 72)
    yaz(b)
    yaz("=" * 72)


def profil_tahmin(ev, seri, bitis):
    """Hafta-gun-saat profili: yalnizca 'bitis'ten once ve PROFIL_BASLANGIC'tan sonra."""
    s = seri[(seri.index < bitis) & (seri.index >= PROFIL_BASLANGIC)]
    d = pd.DataFrame({"v": s.to_numpy(), "wd": s.index.dayofweek, "hr": s.index.hour})
    prof = d.groupby(["wd", "hr"])["v"].mean().rename("profil").reset_index()
    pv = ev[["target_weekday", "target_hour"]].merge(
        prof, left_on=["target_weekday", "target_hour"], right_on=["wd", "hr"], how="left"
    )["profil"].to_numpy()
    return pv, float(s.mean())


def b0(ev, seri, bitis):
    pv, genel = profil_tahmin(ev, seri, bitis)
    return pv / genel * ev[SEVIYE].to_numpy()


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
        "zirve_gercek_tahmin_orani": float(a24[gp].mean() / max(p24[gp].mean(), 1e-9)),
    }


def pinball(a, u, q):
    d = a - u
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def hedef_degerler(seri, tarihler):
    v = seri.reindex(pd.DatetimeIndex(tarihler)).to_numpy()
    if np.isnan(v).any():
        raise SystemExit("Yuk serisinde eksik saat var; saatlik_yaklasik_yuk.csv kontrol edilmeli.")
    return v


# ---------------------------------------------------------------- veri
kol = ["origin_datetime", "target_datetime", "forecast_horizon",
       "target_weekday", "target_hour", SEVIYE, "started_session_energy_kwh"]
train = pd.read_csv(FE / "train_verisi.csv", usecols=kol,
                    parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", usecols=kol,
                  parse_dates=["origin_datetime", "target_datetime"])
tum = pd.concat([train, val], ignore_index=True)
assert tum.target_datetime.max() < TEST_BASLANGIC, "Test verisi karismis olabilir!"

yuk = pd.read_csv(LP, encoding="utf-8-sig", parse_dates=["datetime"], index_col="datetime")
yuk = yuk[yuk.index < TEST_BASLANGIC]
bas_seri = (tum.drop_duplicates("target_datetime")
            .set_index("target_datetime")["started_session_energy_kwh"].sort_index())

yaz("Yuk senaryolari icin B0 (TEST VERISI KULLANILMAMISTIR)")
yaz(f"Yuk serisi: {yuk.index.min()} - {yuk.index.max()} ({len(yuk)} saat)")
yaz(f"Profil/kalibrasyon kesme boslugu: {BOSLUK}; kalibrasyon penceresi: {KAL_HAFTA} hafta")
yaz("Seviye: origin aninda bilinen BASLANGIC enerjisi serisinin son 168 saat ortalamasi.")

sonuc, sinir = [], []
for sc, kolon in SERILER.items():
    seri = yuk[kolon]
    for bas, bit in FOLDLAR:
        bas_t, bit_t = pd.Timestamp(bas), pd.Timestamp(bit)
        kesme = bas_t - BOSLUK
        cal_bas = kesme - pd.Timedelta(weeks=KAL_HAFTA)
        fold_ad = bas_t.strftime("%Y-%m")

        ev = tum[(tum.target_datetime >= bas_t) & (tum.target_datetime < bit_t)]
        cal = tum[(tum.target_datetime >= cal_bas) & (tum.target_datetime < kesme)]
        a_ev = hedef_degerler(seri, ev.target_datetime)
        a_cal = hedef_degerler(seri, cal.target_datetime)
        h24 = (ev["forecast_horizon"].to_numpy() == 24)

        pv, _ = profil_tahmin(ev, seri, kesme)
        b_ev = b0(ev, seri, kesme)
        tahmin = {
            "B0_yuk_serisi": b_ev,
            "Profil_statik_yuk": pv,
            "Naif_baslangic_tahmini_yuk_sanildi": b0(ev, bas_seri, kesme),
        }
        for ad, p in tahmin.items():
            p = np.clip(np.asarray(p, dtype=float), 0, None)
            satir = {"senaryo": sc, "fold": fold_ad, "yontem": ad}
            satir.update(metrik(a_ev, p, h24))
            sonuc.append(satir)

        # ust sinir (S1: karekok olcek), kalibrasyon yalnizca kesme oncesi
        b_cal = np.clip(b0(cal, seri, cal_bas), 0, None)
        b_evc = np.clip(b_ev, 0, None)
        m = float(a_cal.mean())
        ol_cal = (b_cal + 0.05 * m) ** 0.5
        ol_ev = (b_evc + 0.05 * m) ** 0.5
        z = (a_cal - b_cal) / ol_cal
        zirve = b_evc >= np.quantile(b_evc, 0.90)
        for q in QS:
            u = np.clip(b_evc + float(np.quantile(z, q)) * ol_ev, 0, None)
            sinir.append({
                "senaryo": sc, "fold": fold_ad, "q": q,
                "kapsama": float(np.mean(a_ev <= u)),
                "kapsama_zirve": float(np.mean(a_ev[zirve] <= u[zirve])),
                "kapsama_h24": float(np.mean(a_ev[h24] <= u[h24])),
                "pinball": pinball(a_ev, u, q),
                "genislik": float(np.mean(u - b_evc)),
            })
        yaz(f"  tamamlandi: {sc} / {fold_ad}")

df = pd.DataFrame(sonuc)
bd = pd.DataFrame(sinir)
df.to_csv(OUT / "yuk_tahmin_sonuclari.csv", index=False, encoding="utf-8-sig")
bd.to_csv(OUT / "yuk_ust_sinir_sonuclari.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- ozet
for sc in SERILER:
    d = df[df.senaryo == sc]
    s = bd[bd.senaryo == sc]
    bolum(f"SENARYO: {sc}  (birim: kWh/saat)")
    for metr in ["MAE", "WAPE", "bias", "zirve_recall_h24", "zirve_gercek_tahmin_orani"]:
        pv_ = d.pivot(index="yontem", columns="fold", values=metr)
        pv_["ortalama"] = pv_.mean(axis=1)
        pv_ = pv_.sort_values("ortalama") if metr in ("MAE", "WAPE") else pv_
        yaz(f"\n[{metr}]")
        yaz(pv_.round(3).to_string())

    mae_pv = d.pivot(index="yontem", columns="fold", values="MAE")
    b0m, stm = mae_pv.loc["B0_yuk_serisi"], mae_pv.loc["Profil_statik_yuk"]
    nfm = mae_pv.loc["Naif_baslangic_tahmini_yuk_sanildi"]
    k1 = bool(b0m.mean() < stm.mean() and int((b0m < stm).sum()) >= 2)
    yaz(f"\nK1 (B0 < statik profil: ortalama MAE dusuk ve >=2/3 kivrim): "
        f"{b0m.mean():.2f} vs {stm.mean():.2f}; {int((b0m < stm).sum())}/3 -> {'PASS' if k1 else 'FAIL'}")
    yaz(f"Bilgi: B0 yuk tahmini MAE {b0m.mean():.2f}; naif (baslangic tahmini yuk sanilsa) MAE {nfm.mean():.2f}")

    s90 = s[s.q == 0.90]
    k3 = bool(0.85 <= s90.kapsama.mean() <= 0.95 and s90.kapsama.min() >= 0.85)
    yaz(f"\n[Ust sinir q=0.90, kivrim bazinda]")
    yaz(s90[["fold", "kapsama", "kapsama_zirve", "kapsama_h24", "pinball", "genislik"]].round(3).to_string(index=False))
    yaz("\n[Ust sinir, kivrimlar uzerinden ortalama]")
    yaz(s.groupby("q")[["kapsama", "kapsama_zirve", "kapsama_h24", "pinball", "genislik"]].mean().round(3).to_string())
    yaz(f"K3 (q=0.90 kapsama ort. 0.85-0.95 ve hicbir kivrimda <0.85): "
        f"ort={s90.kapsama.mean():.3f}, min={s90.kapsama.min():.3f} -> {'PASS' if k3 else 'FAIL'}")

yol = OUT / "yuk_tahmin_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")