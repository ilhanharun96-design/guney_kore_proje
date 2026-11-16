from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
OUT = BASE / "outputs_modeling" / "belirsizlik_denemeleri"
OUT.mkdir(parents=True, exist_ok=True)

HEDEFLER = ["session_start_count", "started_session_energy_kwh"]
FOLDLAR = [("2022-06-01", "2022-07-01"),
           ("2022-07-01", "2022-08-01"),
           ("2022-08-01", "2022-09-01")]
KALIBRASYON_HAFTA = 8
QS = [0.80, 0.90, 0.95]
OLCEKLER = {"S1_karekok": 0.5, "S2_orantili": 1.0}

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def baseline(tr, ev, hedef):
    """B0: profil (yalnizca tr'den) x son 168 saat seviyesi."""
    tr_u = tr.drop_duplicates("target_datetime")
    prof = (tr_u.groupby(["target_weekday", "target_hour"])[hedef]
            .mean().rename("profil").reset_index())
    genel = tr_u[hedef].mean()
    pv = (ev[["target_weekday", "target_hour"]]
          .merge(prof, on=["target_weekday", "target_hour"], how="left")["profil"].to_numpy())
    return pv / genel * ev[f"{hedef}_rolling_mean_168"].to_numpy()


def pinball(a, u, q):
    d = a - u
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


# ---------------------------------------------------------------- veri
train = pd.read_csv(FE / "train_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", parse_dates=["origin_datetime", "target_datetime"])
tum = pd.concat([train, val], ignore_index=True)
assert tum.target_datetime.max() < pd.Timestamp("2022-09-01"), "Test verisi karismis olabilir!"

yaz("Belirsizlik (ust sinir) denemeleri (TEST VERISI KULLANILMAMISTIR)")
yaz(f"Kalibrasyon penceresi: kivrimdan onceki {KALIBRASYON_HAFTA} hafta; q degerleri: {QS}")

kayit = []
for hedef in HEDEFLER:
    for bas, bit in FOLDLAR:
        bas_t, bit_t = pd.Timestamp(bas), pd.Timestamp(bit)
        c0 = bas_t - pd.Timedelta(weeks=KALIBRASYON_HAFTA)
        fold_ad = bas_t.strftime("%Y-%m")

        onceki = tum[tum.target_datetime < c0]
        cal = tum[(tum.target_datetime >= c0) & (tum.target_datetime < bas_t)]
        egitim = tum[tum.target_datetime < bas_t]
        ev = tum[(tum.target_datetime >= bas_t) & (tum.target_datetime < bit_t)]

        a_cal = cal[hedef].to_numpy()
        b_cal = np.clip(baseline(onceki, cal, hedef), 0, None)
        a_ev = ev[hedef].to_numpy()
        b_ev = np.clip(baseline(egitim, ev, hedef), 0, None)
        m = float(a_cal.mean())
        h24 = (ev["forecast_horizon"].to_numpy() == 24)
        zirve = b_ev >= np.quantile(b_ev, 0.90)

        for ad, p in OLCEKLER.items():
            ol_cal = (b_cal + 0.05 * m) ** p
            ol_ev = (b_ev + 0.05 * m) ** p
            z = (a_cal - b_cal) / ol_cal
            for q in QS:
                zq = float(np.quantile(z, q))
                u = np.clip(b_ev + zq * ol_ev, 0, None)
                kayit.append({
                    "hedef": hedef, "fold": fold_ad, "olcek": ad, "q": q,
                    "kapsama": float(np.mean(a_ev <= u)),
                    "kapsama_zirve": float(np.mean(a_ev[zirve] <= u[zirve])),
                    "kapsama_h24": float(np.mean(a_ev[h24] <= u[h24])),
                    "pinball": pinball(a_ev, u, q),
                    "genislik": float(np.mean(u - b_ev)),
                    "kal_n": len(cal),
                })

df = pd.DataFrame(kayit)
df.to_csv(OUT / "belirsizlik_sonuclari.csv", index=False, encoding="utf-8-sig")

for hedef in HEDEFLER:
    yaz()
    yaz("=" * 72)
    yaz(f"HEDEF: {hedef}")
    yaz("=" * 72)
    skorlar = {}
    for ad in OLCEKLER:
        s = df[(df.hedef == hedef) & (df.olcek == ad)]
        yaz(f"\n[{ad}] q=0.90 kivrim bazinda")
        yaz(s[s.q == 0.90][["fold", "kapsama", "kapsama_zirve", "kapsama_h24", "pinball", "genislik"]]
            .round(3).to_string(index=False))
        ort = s.groupby("q")[["kapsama", "kapsama_zirve", "kapsama_h24", "pinball", "genislik"]].mean()
        yaz(f"\n[{ad}] kivrimlar uzerinden ortalama (her q icin)")
        yaz(ort.round(3).to_string())
        sapma = float(np.mean(np.abs(s.kapsama - s.q)) + np.mean(np.abs(s.kapsama_zirve - s.q))) / 2
        skorlar[ad] = sapma
        yaz(f"  Nominalden ortalama mutlak sapma (genel + zirve): {sapma:.4f}")
    en_iyi = min(skorlar, key=skorlar.get)
    yaz(f"\nOnceden belirlenen kurala gore secilen olcek: {en_iyi}")

yol = OUT / "belirsizlik_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")