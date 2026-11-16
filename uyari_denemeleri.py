from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
LP = BASE / "outputs_load_profile" / "saatlik_yaklasik_yuk.csv"
OUT = BASE / "outputs_load_profile" / "uyari_denemeleri"
OUT.mkdir(parents=True, exist_ok=True)

TEST_BASLANGIC = pd.Timestamp("2022-09-01")
PROFIL_BASLANGIC = pd.Timestamp("2021-10-07 18:00")
REF_BAS = pd.Timestamp("2021-10-07 18:00")
REF_BIT = pd.Timestamp("2022-08-01 00:00")
SERILER = {"U100": "yuk_U100_kwh", "F50": "yuk_F50_kwh", "F25": "yuk_F25_kwh"}
BEKLENEN_REF = {"U100": 329.8, "F50": 386.2, "F25": 429.7}
BEKLENEN_KAPSAMA_U100 = {"2022-06": 0.885, "2022-07": 0.875, "2022-08": 0.914}
FOLDLAR = [("2022-06-01", "2022-07-01"),
           ("2022-07-01", "2022-08-01"),
           ("2022-08-01", "2022-09-01")]
BOSLUK = pd.Timedelta(days=2)
KAL_HAFTA = 8
KAPASITE_ORANLARI = [0.6, 0.7, 0.8, 0.9]
P_ESIKLERI = [0.05, 0.10, 0.25, 0.50]
VARSAYILAN_KURAL = "Olasilik>=0.25"
ASAMA_UFUKLARI = [1, 6, 12, 24]
SEVIYE = "started_session_energy_kwh_rolling_mean_168"
KURAL_SIRASI = (["B0_nokta"] + [f"Olasilik>={p:.2f}" for p in P_ESIKLERI]
                + ["Persistans_168", "Profil_statik", "Sabit_pencere_17_22"])
BASELINE_KURALLAR = ["Persistans_168", "Profil_statik", "Sabit_pencere_17_22"]
GUN_KURALLARI = ["B0_nokta", VARSAYILAN_KURAL, "Persistans_168"]

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def bolum(b):
    yaz()
    yaz("=" * 76)
    yaz(b)
    yaz("=" * 76)


def profil_tahmin(ev, seri, bitis):
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


def hedef_degerler(seri, tarihler):
    v = seri.reindex(pd.DatetimeIndex(tarihler)).to_numpy()
    if np.isnan(v).any():
        raise SystemExit("Yuk serisinde eksik saat var; saatlik_yaklasik_yuk.csv kontrol edilmeli.")
    return v


def sayim(y, al):
    tp = int((y & al).sum())
    fp = int((~y & al).sum())
    fn = int((y & ~al).sum())
    tn = int((~y & ~al).sum())
    return tp, fp, fn, tn


def oranlar(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = np.nan
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ---------------------------------------------------------------- veri
kol = ["origin_datetime", "target_datetime", "forecast_horizon",
       "target_weekday", "target_hour", SEVIYE]
train = pd.read_csv(FE / "train_verisi.csv", usecols=kol,
                    parse_dates=["origin_datetime", "target_datetime"])
val = pd.read_csv(FE / "validation_verisi.csv", usecols=kol,
                  parse_dates=["origin_datetime", "target_datetime"])
tum = pd.concat([train, val], ignore_index=True)
assert tum.target_datetime.max() < TEST_BASLANGIC, "Test verisi karismis olabilir!"

yuk = pd.read_csv(LP, encoding="utf-8-sig", parse_dates=["datetime"], index_col="datetime")
yuk = yuk[yuk.index < TEST_BASLANGIC]

yaz("Kapasite asimi uyari deneyi (TEST VERISI KULLANILMAMISTIR)")
yaz(f"Yuk serisi: {yuk.index.min()} - {yuk.index.max()} ({len(yuk)} saat)")
yaz("Olay: hedef saatte kontrolsuz SAATLIK ORTALAMA yuk (kW) > kapasite. Uyari: hedeften 24 saat once (h=24).")
yaz(f"Varsayilan kural (onceden secildi): {VARSAYILAN_KURAL}")

bolum("0. KONTROLLER")
refler = {}
for sc, kolon in SERILER.items():
    seri = yuk[kolon]
    ref = float(seri[(seri.index >= REF_BAS) & (seri.index < REF_BIT)].quantile(0.99))
    refler[sc] = ref
    durum = "GECTI" if abs(ref - BEKLENEN_REF[sc]) < 0.1 else "UYARI: simulasyondaki referanstan farkli"
    yaz(f"  Referans {sc}: {ref:.1f} kW (beklenen {BEKLENEN_REF[sc]}) -> {durum}")

kayit, gun_kayit, asama_kayit = [], [], []
skorlar = {}
kapsama_kontrol = []

for sc, kolon in SERILER.items():
    seri = yuk[kolon]
    ref = refler[sc]
    for bas, bit in FOLDLAR:
        bas_t, bit_t = pd.Timestamp(bas), pd.Timestamp(bit)
        kesme = bas_t - BOSLUK
        cal_bas = kesme - pd.Timedelta(weeks=KAL_HAFTA)
        fold_ad = bas_t.strftime("%Y-%m")

        ev = tum[(tum.target_datetime >= bas_t) & (tum.target_datetime < bit_t)].reset_index(drop=True)
        cal = tum[(tum.target_datetime >= cal_bas) & (tum.target_datetime < kesme)]
        a_ev = hedef_degerler(seri, ev.target_datetime)
        a_cal = hedef_degerler(seri, cal.target_datetime)
        h = ev["forecast_horizon"].to_numpy()
        h24 = (h == 24)

        pv, _ = profil_tahmin(ev, seri, kesme)
        b_ev = np.clip(b0(ev, seri, kesme), 0, None)
        b_cal = np.clip(b0(cal, seri, cal_bas), 0, None)
        m = float(a_cal.mean())
        ol_cal = (b_cal + 0.05 * m) ** 0.5
        ol_ev = (b_ev + 0.05 * m) ** 0.5
        z = np.sort((a_cal - b_cal) / ol_cal)
        pers = hedef_degerler(seri, ev.target_datetime - pd.Timedelta(hours=168))
        saat = ev["target_hour"].to_numpy()
        tarih = ev["target_datetime"].dt.normalize().to_numpy()

        if sc == "U100":
            u90 = np.clip(b_ev + float(np.quantile(z, 0.90)) * ol_ev, 0, None)
            kap90 = float(np.mean(a_ev[h24] <= u90[h24]))
            kapsama_kontrol.append((fold_ad, kap90, BEKLENEN_KAPSAMA_U100[fold_ad]))

        for kap in KAPASITE_ORANLARI:
            cap = kap * ref
            y = a_ev > cap
            thr = (cap - b_ev) / ol_ev
            p = 1.0 - np.searchsorted(z, thr, side="right") / len(z)

            kurallar = {"B0_nokta": b_ev > cap}
            for ps in P_ESIKLERI:
                kurallar[f"Olasilik>={ps:.2f}"] = p >= ps
            kurallar["Persistans_168"] = pers > cap
            kurallar["Profil_statik"] = pv > cap
            kurallar["Sabit_pencere_17_22"] = (saat >= 17) & (saat <= 22)

            # saat duzeyi (h=24)
            for ad, al in kurallar.items():
                tp, fp, fn, tn = sayim(y[h24], al[h24])
                _, rec, _ = oranlar(tp, fp, fn)
                kayit.append({"sc": sc, "kap": kap, "cap_kw": cap, "fold": fold_ad, "kural": ad,
                              "N": int(h24.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "recall": rec})

            # gun duzeyi (h=24)
            for ad in GUN_KURALLARI:
                dfg = pd.DataFrame({"d": tarih[h24], "y": y[h24], "al": kurallar[ad][h24]})
                g = dfg.groupby("d").agg(y=("y", "any"), al=("al", "any"))
                tp, fp, fn, tn = sayim(g["y"].to_numpy(), g["al"].to_numpy())
                gun_kayit.append({"sc": sc, "kap": kap, "fold": fold_ad, "kural": ad,
                                  "gun_N": len(g), "tp": tp, "fp": fp, "fn": fn, "tn": tn})

            # uyari zamanlamasi (varsayilan kural)
            for H in ASAMA_UFUKLARI:
                mk = (h == H)
                tp, fp, fn, tn = sayim(y[mk], kurallar[VARSAYILAN_KURAL][mk])
                asama_kayit.append({"sc": sc, "kap": kap, "fold": fold_ad, "ufuk": H,
                                    "tp": tp, "fp": fp, "fn": fn, "tn": tn})

            # esikten bagimsiz siralama icin skorlar
            anahtar = (sc, kap)
            d = skorlar.setdefault(anahtar, {"y": [], "p": [], "b": [], "pv": []})
            d["y"].append(y[h24])
            d["p"].append(p[h24])
            d["b"].append(b_ev[h24])
            d["pv"].append(pv[h24])
        yaz(f"  tamamlandi: {sc} / {fold_ad}")

for fold_ad, kap90, bek in kapsama_kontrol:
    durum = "GECTI" if abs(kap90 - bek) < 0.0015 else "UYARI: onceki deneyden farkli"
    yaz(f"  Kontrol U100 q=0.90 h=24 kapsama {fold_ad}: {kap90:.3f} (onceki deney {bek}) -> {durum}")

res = pd.DataFrame(kayit)
res.to_csv(OUT / "uyari_fold_sonuclari.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- havuzlanmis saat duzeyi
havuz = (res.groupby(["sc", "kap", "kural"], sort=False)
         .agg(cap_kw=("cap_kw", "first"), N=("N", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
              fn=("fn", "sum"), tn=("tn", "sum"), min_fold_recall=("recall", "min"))
         .reset_index())
havuz["olay_%"] = (havuz.tp + havuz.fn) / havuz.N * 100
havuz["uyari_%"] = (havuz.tp + havuz.fp) / havuz.N * 100
havuz["precision"] = havuz.tp / (havuz.tp + havuz.fp).replace(0, np.nan)
havuz["recall"] = havuz.tp / (havuz.tp + havuz.fn).replace(0, np.nan)
havuz["F1"] = 2 * havuz.precision * havuz.recall / (havuz.precision + havuz.recall)
havuz["yanlis_alarm_saat_ay"] = havuz.fp / len(FOLDLAR)
havuz.to_csv(OUT / "uyari_havuzlanmis_sonuclar.csv", index=False, encoding="utf-8-sig")

bolum("1. OLAY SAYILARI (h=24 hedef saatleri; kivrim basina asim saati)")
for sc in SERILER:
    t = res[(res.sc == sc) & (res.kural == "B0_nokta")].copy()
    t["olay"] = t.tp + t.fn
    pv_ = t.pivot(index="kap", columns="fold", values="olay")
    pv_["toplam"] = pv_.sum(axis=1)
    pv_["saat_toplam"] = int(t.groupby("kap")["N"].sum().iloc[0])
    yaz(f"[{sc}]")
    yaz(pv_.astype(int).to_string())

bolum("2. SAAT DUZEYI UYARI SONUCLARI (3 kivrim birlestirildi; uyari 24 saat once)")
sutunlar = ["kural", "olay_%", "uyari_%", "precision", "recall", "F1", "yanlis_alarm_saat_ay", "min_fold_recall"]
for sc in SERILER:
    for kap in KAPASITE_ORANLARI:
        t = havuz[(havuz.sc == sc) & (havuz.kap == kap)].copy()
        t["kural"] = pd.Categorical(t["kural"], KURAL_SIRASI, ordered=True)
        t = t.sort_values("kural")
        yaz(f"\n[{sc}] kapasite %{int(kap * 100)} = {t.cap_kw.iloc[0]:.0f} kW")
        yaz(t[sutunlar].round(3).to_string(index=False))

bolum("3. ESIKTEN BAGIMSIZ SIRALAMA KALITESI (h=24, 3 kivrim birlestirildi)")
sk = []
for (sc, kap), d in skorlar.items():
    y = np.concatenate(d["y"])
    satir = {"sc": sc, "kap": kap, "olay_%": float(y.mean() * 100)}
    if y.sum() > 0 and (~y).sum() > 0:
        for ad, anh in [("p", "p"), ("B0_nokta", "b"), ("Profil_statik", "pv")]:
            s_ = np.concatenate(d[anh])
            satir[f"AP_{ad}"] = average_precision_score(y, s_)
            satir[f"AUC_{ad}"] = roc_auc_score(y, s_)
    sk.append(satir)
skdf = pd.DataFrame(sk)
skdf.to_csv(OUT / "uyari_siralama_kalitesi.csv", index=False, encoding="utf-8-sig")
yaz(skdf.round(3).to_string(index=False))

bolum("4. GUN DUZEYI UYARI (gun = o gunun herhangi bir saati; 3 kivrim birlestirildi)")
gun = pd.DataFrame(gun_kayit)
gh = (gun.groupby(["sc", "kap", "kural"], sort=False)
      .agg(gun_N=("gun_N", "sum"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
      .reset_index())
gh["olay_gun_%"] = (gh.tp + gh.fn) / gh.gun_N * 100
gh["precision"] = gh.tp / (gh.tp + gh.fp).replace(0, np.nan)
gh["recall"] = gh.tp / (gh.tp + gh.fn).replace(0, np.nan)
gh.to_csv(OUT / "uyari_gun_duzeyi.csv", index=False, encoding="utf-8-sig")
for sc in SERILER:
    yaz(f"\n[{sc}]")
    yaz(gh[gh.sc == sc][["kap", "kural", "gun_N", "olay_gun_%", "tp", "fp", "fn", "precision", "recall"]]
        .round(3).to_string(index=False))
yaz("\nNot: Kapasite dusukse neredeyse her gun olay vardir; gun duzeyi uyari yalnizca yuksek kapasitede bilgi verir.")

bolum(f"5. UYARI ZAMANLAMASI (kural: {VARSAYILAN_KURAL}; ufuk = hedeften kac saat once)")
asama = pd.DataFrame(asama_kayit)
ah = asama.groupby(["sc", "kap", "ufuk"], sort=False)[["tp", "fp", "fn", "tn"]].sum().reset_index()
ah["precision"] = ah.tp / (ah.tp + ah.fp).replace(0, np.nan)
ah["recall"] = ah.tp / (ah.tp + ah.fn).replace(0, np.nan)
ah.to_csv(OUT / "uyari_zamanlamasi.csv", index=False, encoding="utf-8-sig")
for sc in SERILER:
    pv_ = ah[ah.sc == sc].pivot(index="kap", columns="ufuk", values="recall")
    pp_ = ah[ah.sc == sc].pivot(index="kap", columns="ufuk", values="precision")
    yaz(f"\n[{sc}] recall (satir: kapasite orani; sutun: ufuk saat)")
    yaz(pv_.round(3).to_string())
    yaz(f"[{sc}] precision")
    yaz(pp_.round(3).to_string())

bolum("6. ONCEDEN YAZILAN OLCUTLER (karar: U100, kapasite %70 ve %80)")
yaz(f"Kural: {VARSAYILAN_KURAL}. C1: F1 > en iyi karsilastirma kuralinin F1'i; "
    f"C2: recall >= 0.80 ve hicbir kivrimda recall < 0.70; C3: precision >= 0.60")
for sc in SERILER:
    tum_gecti = True
    for kap in [0.7, 0.8]:
        t = havuz[(havuz.sc == sc) & (havuz.kap == kap)].set_index("kural")
        v = t.loc[VARSAYILAN_KURAL]
        en_iyi_base = float(np.nanmax([t.loc[k, "F1"] for k in BASELINE_KURALLAR]))
        c1 = bool(v["F1"] > en_iyi_base)
        c2 = bool(v["recall"] >= 0.80 and v["min_fold_recall"] >= 0.70)
        c3 = bool(v["precision"] >= 0.60)
        tum_gecti &= (c1 and c2 and c3)
        yaz(f"  [{sc}] kapasite %{int(kap * 100)}: F1={v['F1']:.3f} (en iyi karsilastirma {en_iyi_base:.3f}) "
            f"C1={'PASS' if c1 else 'FAIL'} | recall={v['recall']:.3f}, min kivrim={v['min_fold_recall']:.3f} "
            f"C2={'PASS' if c2 else 'FAIL'} | precision={v['precision']:.3f} C3={'PASS' if c3 else 'FAIL'}")
    yaz(f"  [{sc}] GENEL: {'PASS' if tum_gecti else 'FAIL'}" + ("   <- karar senaryosu" if sc == "U100" else "   (bilgi amacli)"))

yol = OUT / "uyari_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")