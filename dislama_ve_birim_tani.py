import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(errors="replace")

BASE = Path(__file__).resolve().parent
VERI = BASE / "data" / "ChargingRecords.csv"
OUT = BASE / "outputs_load_profile"
OUT.mkdir(parents=True, exist_ok=True)

RAPOR = {
    "negatif_duration": (17, 464.11),
    "sifir_duration": (718, 3046.02),
    "fark_5dk (rapor tanimi)": (69, 1523.97),
    "baglanti_araligi_le0": (47, 705.90),
}
RAPOR_BIRLESIM = (804, 5034.10)
N_SAAT = 8767

SATIRLAR = []


def yaz(m=""):
    print(m)
    SATIRLAR.append(str(m))


def bolum(b):
    yaz()
    yaz("=" * 72)
    yaz(b)
    yaz("=" * 72)


def oku(yol):
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(yol, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError("CSV okunamadi (kodlama).")


ham = oku(VERI)
df = ham.drop_duplicates().copy()
df["start"] = pd.to_datetime(df["StartDatetime"], errors="coerce")
df["end"] = pd.to_datetime(df["EndDatetime"], errors="coerce")
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
df["aralik_dk"] = (df["end"] - df["start"]).dt.total_seconds() / 60.0
df["fark_dk"] = df["Duration"] - df["aralik_dk"]
toplam_kwh = df["Demand"].sum()

bolum("1. HAM SUTUN ORNEKLERI (bicim kontrolu)")
yaz(f"Tekrarsiz satir: {len(df)}")
yaz(df[["StartDay", "StartTime", "StartDatetime", "EndDay", "EndTime", "EndDatetime",
        "Duration", "Demand"]].head(3).to_string())

bolum("2. DISLAMA NEDENLERI (rapor terimleriyle) - rapordaki sayilarla yan yana")
pozitif = (df["Duration"] > 0) & (df["aralik_dk"] > 0)
nedenler = {
    "negatif_duration": df["Duration"] < 0,
    "sifir_duration": df["Duration"] == 0,
    "fark_5dk (rapor tanimi)": pozitif & (df["fark_dk"].abs() > 5),
    "baglanti_araligi_le0": df["aralik_dk"] <= 0,
}
for ad, mask in nedenler.items():
    r = RAPOR[ad]
    yaz(f"  {ad:28s} bu calisma: {int(mask.sum()):6d} oturum, {df.loc[mask, 'Demand'].sum():10.2f} kWh"
        f"   | rapor: {r[0]} oturum, {r[1]} kWh")
birlesim = np.zeros(len(df), dtype=bool)
for mask in nedenler.values():
    birlesim |= mask.to_numpy()
yaz(f"  BIRLESIM (ayni oturum bir kez): {int(birlesim.sum())} oturum, "
    f"{df.loc[birlesim, 'Demand'].sum():.2f} kWh | rapor: {RAPOR_BIRLESIM[0]} oturum, {RAPOR_BIRLESIM[1]} kWh")
yaz(f"  Eksik/negatif Demand: {int((df['Demand'].isna() | (df['Demand'] < 0)).sum())}")
yaz(f"  Yalnizca Duration>0 ve aralik>0 kosuluyla kalan oturum: {int(pozitif.sum())}")

bolum("3. DURATION - BAGLANTI ARALIGI FARKI (Duration>0 ve aralik>0 olan oturumlar)")
p = df[pozitif].copy()
bant = pd.cut(p["fark_dk"].abs(), [-0.001, 1, 5, 10, 30, 120, np.inf],
              labels=["<=1 dk", "1-5", "5-10", "10-30", "30-120", ">120"])
tab = pd.DataFrame({"oturum": bant.value_counts().sort_index(),
                    "kwh": p.groupby(bant, observed=False)["Demand"].sum()})
yaz(tab.round(1).to_string())
buyuk = p[p["fark_dk"].abs() > 5]
yaz(f"|fark|>5 dk olanlar: Duration < aralik: {int((buyuk['fark_dk'] < 0).sum())}, "
    f"Duration > aralik: {int((buyuk['fark_dk'] > 0).sum())}")
yaz(f"fark_dk yuzdelikleri (p1,p5,p50,p95,p99): " +
    ", ".join(f"{v:.1f}" for v in p["fark_dk"].quantile([0.01, 0.05, 0.5, 0.95, 0.99])))

bolum("4. GUN+SAAT SUTUNLARI ILE DATETIME SUTUNLARI TUTARLI MI?")
for on in ["Start", "End"]:
    g, s = f"{on}Day", f"{on}Time"
    if g in df.columns and s in df.columns:
        olus = pd.to_datetime(df[g].astype(str).str.strip() + " " + df[s].astype(str).str.strip(),
                              errors="coerce")
        ref = df["start"] if on == "Start" else df["end"]
        fark = (olus - ref).dt.total_seconds().abs()
        yaz(f"  {on}: ayristirilamayan = {int(olus.isna().sum())}, "
            f"60 saniyeden fazla farkli = {int((fark > 60).sum())}")

bolum("5. ASIRI ORTALAMA GUC KAYITLARI (Demand / Duration)")
p["kw_ort"] = p["Demand"] / (p["Duration"] / 60.0)
top = p.sort_values("kw_ort", ascending=False).head(15)[
    ["start", "ChargerID", "Duration", "aralik_dk", "Demand", "kw_ort"]]
yaz(top.round(2).to_string(index=False))
top.to_csv(OUT / "asiri_guc_kayitlari.csv", index=False, encoding="utf-8-sig")
yaz(f"150 kW ustu oturum: {int((p['kw_ort'] > 150).sum())}, toplam enerjisi: "
    f"{p.loc[p['kw_ort'] > 150, 'Demand'].sum():.1f} kWh")

bolum("6. LOCATION YAPISI (lokasyon duzeyi tahmin mumkun mu?)")
loc = df["Location"].astype(str)
sayi = loc.value_counts()
yaz(f"Farkli Location: {sayi.size}; farkli ChargerCompany: {df['ChargerCompany'].nunique()}; "
    f"farkli ChargerID: {df['ChargerID'].nunique()}")
yaz("Lokasyon basina oturum sayisi: " +
    ", ".join(f"{ad}={v:.0f}" for ad, v in
              zip(["medyan", "p90", "p99", "max"], sayi.quantile([0.5, 0.9, 0.99, 1.0]))))
for esik, ad in [(N_SAAT, "ortalama >=1 oturum/saat"), (N_SAAT // 2, ">=0,5 oturum/saat"),
                 (N_SAAT // 10, ">=0,1 oturum/saat")]:
    yaz(f"  {ad}: {int((sayi >= esik).sum())} lokasyon")
top10 = sayi.head(10)
chg = df.groupby(loc)["ChargerID"].nunique().reindex(top10.index)
yaz("\nEn cok oturumlu 10 lokasyon:")
yaz(pd.DataFrame({"oturum": top10, "oturum_saat": (top10 / N_SAAT).round(2),
                  "charger_sayisi": chg}).to_string())
yaz(f"Ilk 10 lokasyonun toplam oturum payi: %{top10.sum() / len(df) * 100:.1f}")

yol = OUT / "dislama_tani.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")