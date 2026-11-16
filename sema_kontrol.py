from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
FE = BASE / "outputs_feature_engineering"
MD = BASE / "outputs_modeling"

satirlar = []

def yaz(metin=""):
    print(metin)
    satirlar.append(str(metin))

dosyalar = [
    FE / "train_verisi.csv",
    FE / "validation_verisi.csv",
    FE / "test_verisi.csv",
    FE / "ozellik_listesi.csv",
    MD / "model_karsilastirma_sonuclari.csv",
    MD / "test_tahminleri.csv",
    MD / "ufka_gore_test_hatalari.csv",
]

for yol in dosyalar:
    yaz("=" * 70)
    yaz(f"DOSYA: {yol}")
    if not yol.exists():
        yaz("  -> BULUNAMADI")
        continue
    df = pd.read_csv(yol)
    yaz(f"  Satir: {len(df)}  Sutun: {df.shape[1]}")
    yaz("  Sutunlar ve tipleri:")
    for ad, tip in df.dtypes.items():
        yaz(f"    {ad}: {tip}")
    yaz("  Ilk 2 satir:")
    yaz(df.head(2).to_string())

yaz("=" * 70)
yaz(f"MODELS KLASORU: {MD / 'models'}")
modeller = MD / "models"
if modeller.exists():
    for p in sorted(modeller.rglob("*")):
        if p.is_file():
            yaz(f"  {p.relative_to(modeller)}  ({p.stat().st_size/1024:.1f} KB)")
else:
    yaz("  -> BULUNAMADI")

cikti = MD / "sema_ozeti.txt"
cikti.write_text("\n".join(satirlar), encoding="utf-8")
print(f"\nOzet kaydedildi: {cikti}")