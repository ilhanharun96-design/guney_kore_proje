import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
yol = BASE / "DURUM.md"

if len(sys.argv) < 3:
    raise SystemExit('Kullanim: durum_not_ekle.py "Baslik" "Not metni"')
if not yol.exists():
    raise SystemExit("DURUM.md bulunamadi. Once durum_guncelle.py ile olusturulmali.")

baslik = sys.argv[1]
metin = sys.argv[2]

icerik = yol.read_text(encoding="utf-8")
if "## Güncelleme günlüğü" not in icerik:
    icerik += "\n\n## Güncelleme günlüğü\n"

icerik += f"\n### {datetime.now():%d.%m.%Y %H:%M} - {baslik}\n\n{metin}\n"
yol.write_text(icerik, encoding="utf-8")
print(f"Not eklendi: {baslik}")