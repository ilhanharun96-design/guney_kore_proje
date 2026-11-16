from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
VERI = BASE / "data" / "ChargingRecords.csv"
OUT = BASE / "outputs_load_profile"
OUT.mkdir(parents=True, exist_ok=True)

TEST_BASLANGIC = pd.Timestamp("2022-09-01")
ISINMA_BITIS = pd.Timestamp("2021-10-07")
FRAKSIYONLAR = {"U100": 1.0, "F50": 0.5, "F25": 0.25}

RAPOR = {
    "negatif_duration": (17, 464.11),
    "sifir_duration": (718, 3046.02),
    "duration_baglantidan_5dk_fazla": (69, 1523.97),
    "baglanti_araligi_le0": (47, 705.90),
}
RAPOR_BIRLESIM = (804, 5034.10)
RAPOR_TEKRARSIZ = 72843
RAPOR_GECERLI = 72039
RAPOR_SAAT = 8767

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
    raise RuntimeError("CSV dosyasi okunamadi (kodlama).")


def yuk_profili(s, e, deger, n_toplam):
    """Her oturumun 'deger' miktarini [s, e] araligina (saat cinsinden) esit yayar."""
    yuk = np.zeros(n_toplam)
    for si, ei, vi in zip(s, e, deger):
        d = ei - si
        if d <= 0:
            continue
        p = vi / d
        k0 = int(np.floor(si))
        k1 = int(np.ceil(ei))
        if k1 - k0 <= 1:
            yuk[k0] += vi
        else:
            yuk[k0] += p * (k0 + 1 - si)
            if k1 - k0 > 2:
                yuk[k0 + 1:k1 - 1] += p
            yuk[k1 - 1] += p * (ei - (k1 - 1))
    return yuk


# ------------------------------------------------------------ veri okuma
if not VERI.exists():
    raise SystemExit(f"Dosya bulunamadi: {VERI}")
ham = oku(VERI)
gerekli = ["ChargerID", "StartDay", "StartTime", "EndDay", "EndTime", "Demand", "Duration"]
eksik = [c for c in gerekli if c not in ham.columns]
if eksik:
    raise SystemExit(f"Eksik sutunlar: {eksik}. Mevcut sutunlar: {list(ham.columns)}")

df = ham.drop_duplicates().copy()
bolum("1. VERI OKUMA KONTROLU (orijinal dosya degistirilmez)")
yaz(f"Ham satir: {len(ham)}  Tekrarsiz satir: {len(df)}  (rapor: {RAPOR_TEKRARSIZ})")
if len(df) != RAPOR_TEKRARSIZ:
    raise SystemExit("Tekrarsiz kayit sayisi rapordan farkli; once bu fark incelenmeli.")

# Saniye duzeyinde zaman damgasi: Day + Time sutunlari (Datetime sutunlari dakikaya yuvarlanmis)
df["start"] = pd.to_datetime(df["StartDay"].astype(str).str.strip() + " " + df["StartTime"].astype(str).str.strip(),
                             errors="coerce")
df["end"] = pd.to_datetime(df["EndDay"].astype(str).str.strip() + " " + df["EndTime"].astype(str).str.strip(),
                           errors="coerce")
df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
bozuk = df[["start", "end", "Demand", "Duration"]].isna().any(axis=1)
yaz(f"Ayristirilamayan/eksik deger iceren satir: {int(bozuk.sum())} (rapor: 0)")
df = df[~bozuk].copy()

df["aralik_dk"] = (df["end"] - df["start"]).dt.total_seconds() / 60.0

# Duration birimini veriden tespit et
poz = df[(df["aralik_dk"] > 0) & (df["Duration"] > 0)]
oran = float(np.median(poz["Duration"] / poz["aralik_dk"]))
adaylar = {"dakika": 1.0, "saat": 1.0 / 60.0, "saniye": 60.0}
birim = min(adaylar, key=lambda k: abs(np.log(oran / adaylar[k])))
yaz(f"Duration / baglanti araligi (dk) medyan orani: {oran:.4f}  -> tespit edilen Duration birimi: {birim}")
if abs(np.log(oran / adaylar[birim])) > 0.2:
    raise SystemExit("Duration birimi guvenle tespit edilemedi; oran yukarida yazili, lutfen paylasin.")
carpan = {"dakika": 1.0, "saat": 60.0, "saniye": 1.0 / 60.0}[birim]
df["dur_dk"] = df["Duration"] * carpan
df["fark_dk"] = df["dur_dk"] - df["aralik_dk"]   # pozitif: Duration baglanti araligindan UZUN

# ------------------------------------------------------------ dislama (raporun tanimi)
bolum("2. DISLAMA KONTROLU (saniye duzeyi zaman + tek yonlu fark; rapordaki sayilarla)")
nedenler = {
    "negatif_duration": df["dur_dk"] < 0,
    "sifir_duration": df["dur_dk"] == 0,
    "duration_baglantidan_5dk_fazla": df["fark_dk"] > 5,
    "baglanti_araligi_le0": df["aralik_dk"] <= 0,
}
birlesim = np.zeros(len(df), dtype=bool)
for ad, mask in nedenler.items():
    birlesim |= mask.to_numpy()
    r = RAPOR[ad]
    yaz(f"  {ad:32s} bu calisma: {int(mask.sum()):5d} oturum, {df.loc[mask, 'Demand'].sum():9.2f} kWh"
        f"   | rapor: {r[0]} oturum, {r[1]} kWh")
b_n = int(birlesim.sum())
b_kwh = float(df.loc[birlesim, "Demand"].sum())
yaz(f"  BIRLESIM (ayni oturum bir kez)   bu calisma: {b_n:5d} oturum, {b_kwh:9.2f} kWh"
    f"   | rapor: {RAPOR_BIRLESIM[0]} oturum, {RAPOR_BIRLESIM[1]} kWh")
if b_n == RAPOR_BIRLESIM[0] and abs(b_kwh - RAPOR_BIRLESIM[1]) < 0.01:
    yaz("  SONUC: RAPORLA UYUMLU (dislama tanimi dogrulandi)")
else:
    yaz("  UYARI: Rapordan farkli. Asagidaki sonuclar bu fark bilinerek yorumlanmali; lutfen bu bolumu paylasin.")

gecerli = df[~birlesim].copy()
yaz(f"Yaklasik guc/yuk analizine uygun oturum: {len(gecerli)}  (rapor: {RAPOR_GECERLI})")
kisa = gecerli[gecerli["fark_dk"] < -5]
yaz(f"Bilgi: Duration baglanti araligindan 5 dk'dan fazla KISA olan ve DAHIL edilen oturum: {len(kisa)}, "
    f"{kisa['Demand'].sum():.1f} kWh")
yaz("  Varsayim: bu oturumlarda enerji oturum basindan itibaren Duration suresince verilmis kabul edilir.")

# ------------------------------------------------------------ oturum ozellikleri
gec = gecerli[(gecerli["start"] >= ISINMA_BITIS) & (gecerli["start"] < TEST_BASLANGIC)].copy()
bolum("3. OTURUM OZELLIKLERI (yalnizca 07.10.2021 - 31.08.2022 baslayan gecerli oturumlar)")
yaz(f"Oturum sayisi: {len(gec)}")
dq = gec["dur_dk"].quantile([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).round(1)
yaz("Sure (dakika) yuzdelikleri: " + ", ".join(f"p{int(k*100)}={v}" for k, v in dq.items()) +
    f", max={gec['dur_dk'].max():.1f}")
for esik in [60, 240, 720, 1440]:
    yaz(f"  Suresi > {esik} dk olan oturum orani: %{(gec['dur_dk'] > esik).mean() * 100:.2f}")
gec["kw_ort"] = gec["Demand"] / (gec["dur_dk"] / 60.0)
kq = gec["kw_ort"].quantile([0.05, 0.25, 0.50, 0.75, 0.95, 0.99]).round(2)
yaz("Oturum ortalama gucu (Demand/Duration, kW): " + ", ".join(f"p{int(k*100)}={v}" for k, v in kq.items()) +
    f", max={gec['kw_ort'].max():.1f}")
yaz(f"  Ortalama guc > 50 kW: {int((gec['kw_ort'] > 50).sum())}; > 150 kW: {int((gec['kw_ort'] > 150).sum())}; "
    f"> 350 kW: {int((gec['kw_ort'] > 350).sum())} oturum")
yaz("Not: Bunlar oturum ortalamasidir; anlik guc degildir. Gucu asiri yuksek kayitlar dislanmadi (raporun karari).")

# ------------------------------------------------------------ yuk profili
t0 = df["start"].min().floor("h")
n_grid = int((df["start"].max().floor("h") - t0) / pd.Timedelta(hours=1)) + 1
yaz(f"\nSaatlik izgara: {t0} baslangic, {n_grid} saat (rapor: {RAPOR_SAAT})")

s = ((gecerli["start"] - t0).dt.total_seconds() / 3600.0).to_numpy()
d_h = (gecerli["dur_dk"] / 60.0).to_numpy()
kwh = gecerli["Demand"].to_numpy()
n_toplam = int(np.ceil((s + d_h).max())) + 3

idx = pd.date_range(t0, periods=n_grid, freq="h")
sonuc = pd.DataFrame(index=idx)
k0 = np.floor(s).astype(int)
sonuc["baslangic_atfi_kwh"] = np.bincount(k0, weights=kwh, minlength=n_toplam)[:n_grid]

bolum("4. YUK PROFILI OLUSTURMA (enerji korunumu kontrolu)")
for ad, f in FRAKSIYONLAR.items():
    yuk = yuk_profili(s, s + f * d_h, kwh, n_toplam)
    fark = abs(yuk.sum() - kwh.sum())
    yaz(f"  {ad}: dagitilan toplam enerji farki = {fark:.6f} kWh (0'a yakin olmali)")
    sonuc[f"yuk_{ad}_kwh"] = yuk[:n_grid]

akt = yuk_profili(s, s + d_h, d_h, n_toplam)
sonuc["aktif_oturum_ort"] = akt[:n_grid]   # Duration suresince aktif sayilan oturum sayisi
sonuc["donem"] = np.where(sonuc.index < TEST_BASLANGIC, "egitim_dogrulama", "test")
sonuc.index.name = "datetime"
sonuc.to_csv(OUT / "saatlik_yaklasik_yuk.csv", encoding="utf-8-sig")

# ------------------------------------------------------------ istatistikler
maske = (sonuc.index >= ISINMA_BITIS) & (sonuc.index < TEST_BASLANGIC)
st = sonuc[maske]
seriler = ["baslangic_atfi_kwh", "yuk_U100_kwh", "yuk_F50_kwh", "yuk_F25_kwh"]

bolum("5. SENARYO KARSILASTIRMASI (saatlik, 07.10.2021 - 31.08.2022; kWh/saat = ortalama kW)")
tab = pd.DataFrame({
    "ortalama": st[seriler].mean(),
    "medyan": st[seriler].median(),
    "p95": st[seriler].quantile(0.95),
    "p99": st[seriler].quantile(0.99),
    "maksimum": st[seriler].max(),
    "korelasyon_baslangic_atfi": st[seriler].corr()["baslangic_atfi_kwh"],
})
yaz(tab.round(2).to_string())
yaz(f"\nDuration suresince aktif oturum sayisi (saatlik ortalama): ort={st['aktif_oturum_ort'].mean():.2f}, "
    f"p95={st['aktif_oturum_ort'].quantile(0.95):.2f}, max={st['aktif_oturum_ort'].max():.2f}")

bolum("6. ENERJININ BASLANGIC SAATINDEN SONRAYA TASINAN PAYI")
s2 = ((gec["start"] - t0).dt.total_seconds() / 3600.0).to_numpy()
d2 = (gec["dur_dk"] / 60.0).to_numpy()
k2 = np.floor(s2)
for ad, f in FRAKSIYONLAR.items():
    ilk = np.minimum(1.0, (k2 + 1 - s2) / (f * d2)) * gec["Demand"].to_numpy()
    yaz(f"  {ad}: enerjinin baslangic saatinden SONRAKI saatlere dusen payi = %{(1 - ilk.sum() / gec['Demand'].sum()) * 100:.1f}")

bolum("7. GUN ICI ORTALAMA PROFIL (saat bazinda ortalama, kWh/saat)")
saat = st.groupby(st.index.hour)[seriler + ["aktif_oturum_ort"]].mean()
yaz(saat.round(2).to_string())
saat.to_csv(OUT / "gun_ici_ortalama_profil.csv", encoding="utf-8-sig")

# ------------------------------------------------------------ grafikler
fig, ax = plt.subplots(figsize=(10, 5))
etiket = {"baslangic_atfi_kwh": "Baslangic saatine atfedilen (mevcut hedef)",
          "yuk_F25_kwh": "Yaklasik yuk F25 (ilk %25'te)",
          "yuk_F50_kwh": "Yaklasik yuk F50 (ilk %50'de)",
          "yuk_U100_kwh": "Yaklasik yuk U100 (duzgun)"}
for k in seriler:
    ax.plot(saat.index, saat[k], marker="o", label=etiket[k])
ax.set_xlabel("Saat")
ax.set_ylabel("kWh / saat (ortalama kW)")
ax.set_title("Gun ici ortalama profil (07.10.2021-31.08.2022)")
ax.set_xticks(range(0, 24, 2))
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "01_gun_ici_profil.png", dpi=150)
plt.close(fig)

zirve_t = st["yuk_U100_kwh"].idxmax()
pen = sonuc.loc[zirve_t - pd.Timedelta(days=3, hours=12): zirve_t + pd.Timedelta(days=3, hours=12)]
fig, ax = plt.subplots(figsize=(11, 5))
for k in ["baslangic_atfi_kwh", "yuk_F25_kwh", "yuk_U100_kwh"]:
    ax.plot(pen.index, pen[k], label=etiket[k])
ax.set_ylabel("kWh / saat (ortalama kW)")
ax.set_title(f"En yuksek U100 yukunun bulundugu hafta (zirve: {zirve_t})")
ax.grid(alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "02_zirve_hafta.png", dpi=150)
plt.close(fig)

yaz(f"\nGrafikler ve CSV dosyalari kaydedildi: {OUT}")
yol = OUT / "yaklasik_yuk_ozeti.txt"
yol.write_text("\n".join(SATIRLAR), encoding="utf-8")
print(f"\nOzet kaydedildi: {yol}")