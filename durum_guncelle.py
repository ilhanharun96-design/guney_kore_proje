from datetime import datetime
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
yol = BASE / "DURUM.md"

if yol.exists():
    yedek = BASE / f"DURUM_yedek_{datetime.now():%Y%m%d_%H%M}.md"
    shutil.copy2(yol, yedek)
    print(f"Mevcut DURUM.md yedeklendi: {yedek.name}")

baslik = f"# Proje Durum Dosyası\n\nSon güncelleme: {datetime.now():%d.%m.%Y %H:%M}\n\n"

icerik = r"""
## 1. Proje ve değişmez kurallar

Konu: Elektrikli araç şarj istasyonlarında talep tahmini ve akıllı şarj yönetimi (kısıtlı güç kapasitesi altında karar destek sistemi).

- Orijinal `data\ChargingRecords.csv` asla değiştirilmez.
- `started_session_energy_kwh` gerçek şebeke yükü değildir; oturumun başladığı saate atfedilen toplam enerjidir.
- `ChargerType` 0/1 değerleri resmi veri sözlüğü olmadan AC/DC diye yorumlanmaz.
- Test verisi (Eylül 2022) model geliştirmede kullanılmaz. Yalnızca en sonda, seçilen yöntemle bir kez kullanılacak.
- Yapılmamış iş yapılmış gibi gösterilmez; çıkarımlar "çıkarım" diye etiketlenir.

## 2. Ortam

- Ana klasör: `C:\Users\HARUN\Desktop\ANA_proje\guney_kore_proje\`
- Python: `C:\Users\HARUN\AppData\Local\Programs\Python\Python313\python.exe`
- Veri sütunları (ham CSV): UserID, ChargerID, ChargerCompany, Location, ChargerType, StartDay, StartTime, EndDay, EndTime, StartDatetime, EndDatetime, Duration, Demand
- Saniye düzeyinde zaman için StartDay+StartTime ve EndDay+EndTime kullanılır. StartDatetime/EndDatetime dakikaya yuvarlanmıştır.

## 3. Veri hazırlığı (tamamlandı)

- 72.856 ham kayıt, 13 tekrar çıkarıldı, 72.843 oturum. 2.119 ChargerID. 8.767 saatlik seri (30.09.2021 17:00 – 30.09.2022 23:00).
- Yaklaşık güç analizi için 72.039 oturum uygun; 804 oturum (5.034,10 kWh, toplamın yaklaşık %0,40'ı) dışlandı. Dışlama kuralı yeniden hesaplanıp rapordaki sayılarla birebir doğrulandı.
- Dışlama kuralları: negatif Duration (17), sıfır Duration (718), Duration bağlantı aralığından 5 dakikadan FAZLA uzun (69), bağlantı aralığı <= 0 (47); 47 kayıt birden fazla koşulda.
- Bu 804 kayıt yalnızca güç/yük analizinden dışlandı; oturum sayısı ve başlangıç enerjisi hedeflerinde kalıyor.
- Önemli: Duration bağlantı aralığından 5 dakikadan fazla KISA olan 1.964 oturum (40.017,9 kWh) yük analizine dahil. Enerjileri oturum başından Duration süresince verilmiş kabul edildi (varsayım).
- Özellik mühendisliği: 45 özellik, doğrudan çok adımlı 24 saatlik tahmin. Eğitim 07.10.2021–31.07.2022, doğrulama Ağustos 2022, test Eylül 2022.

## 4. İlk model (referans olarak saklanıyor)

HistGradientBoosting (45 özellik). Eylül 2022 test sonuçları (değişmedi):
- Oturum sayısı: MAE 2,9489; WAPE %34,03
- Başlangıç enerjisi: MAE 61,5374 kWh; WAPE %39,88

Rapordaki "ML modelleri baseline'lardan belirgin iyi" sonucu zayıf baseline'lara (Lag0, Lag168) dayanıyordu. Güçlü baseline ile bu sonuç geçerli değil (aşağıya bakın).

## 5. İlk modelin tanısı (doğrulama Ağustos 2022, test kullanılmadı)

- Doğrulamada model her iki hedefte yaklaşık %21–23 düşük tahmin etti (oturum: gerçek 10,86, tahmin 8,40; enerji: gerçek 192,26, tahmin 152,58 kWh).
- Gecikmeli özellikler ve forecast_horizon için permutation önemi 0; model fiilen "hedef saat" tablosu gibi çalışıyor. Hata ufukla değişmiyor, çünkü ufuk bilgisi kullanılmıyor.
- Eğitimde Ağustos ve Eylül yok; talep artış trendi altında ağaç modeli eğitim aralığının dışına çıkamıyor.
- Aylık seviye eğitim içinde neredeyse mükemmel yeniden üretiliyor (hata ±0,07 oturum), doğrulamada değil.

## 6. Rolling-origin karşılaştırma (Haziran, Temmuz, Ağustos 2022; her kıvrım yalnızca öncesiyle eğitilir)

Ortalama MAE:

| Yöntem | Oturum | Enerji (kWh) |
|---|---|---|
| B0: Profil x Seviye | 2,563 | 55,01 |
| B2, B1, R1, B3 (B0 varyantları) | 2,565 - 2,576 | 55,17 - 56,07 |
| E2: oran hedefli HGB | 2,844 | 60,43 |
| E1: takvim indeksleri çıkarılmış HGB | 2,872 | 61,10 |
| Profil statik | 3,194 | 64,15 |
| REF: ilk model | 3,253 | 71,45 |

- B0 = (eğitimden hafta-gün-saat ortalama profili / profilin genel ortalaması) x son 168 saatin ortalaması. Ayarlanan parametre yok.
- B1, B2, B3 ve R1 (B0 üzerine artık öğrenen HGB), önceden yazılan kurala göre (ortalama MAE düşük ve en az 2/3 kıvrımda iyi) B0'ı geçemedi.
- Oturum sayısında B0, kaba Poisson tabanının yaklaşık %8-12 üstünde.
- REF'in hatası, kıvrımda talep seviyesinin eğitim maksimumunu aştığı oranla birlikte arttı (Temmuz: %86).

## 7. Belirsizlik (üst sınır)

B0 etrafında, son 8 haftanın ölçekli hatalarından üst sınır. Seçilen ölçek: S1 (karekök). q=0,90 kapsama: oturum 0,905, enerji 0,904; zirve saatlerinde 0,941 ve 0,959 (muhafazakâr). Ortalama sınır genişliği: 4,45 oturum, 97,9 kWh.

## 8. Yaklaşık yük profili (07.10.2021 - 31.08.2022 istatistikleri)

Oturum başına enerji, oturumun ilk f kesrinde sabit güçle verilmiş sayılır. Senaryolar: U100 (f=1), F50 (f=0,5), F25 (f=0,25). Bunlar veriden öğrenilmiş değil, duyarlılık varsayımlarıdır.

| Seri | Zirve saati | Zirve (kWh/saat) | Saatlik maks. | Başlangıçtan sonraki saatlere kayan enerji |
|---|---|---|---|---|
| Başlangıca atfedilen | 18:00 | 326,9 | 745,6 | - |
| F25 | 19:00 | 301,2 | 587,1 | %40,7 |
| F50 | 19:00 | 281,8 | 540,3 | %58,8 |
| U100 | 20:00 | 240,6 | 442,6 | %73,3 |

- Oturum ortalama gücü (Demand/Duration): medyan 6,7 kW, çeyrekler 6,08 - 7,07 kW, p95 41,4 kW; 1.055 oturum > 50 kW; 16 oturum > 150 kW (veri hatası olasılığı; raporun kararıyla dışlanmadı).
- Medyan süre 114 dakika; oturumların %18,5'i 4 saatten uzun.
- Senaryo seçimi veriyle yapılamaz. Çıkarım (kanıt değil): ortalama gücün sıkı kümelenmesi U100'ü merkez, F25'i stres senaryosu yapmayı destekler.

## 9. Alınan kararlar

- Tahmin motoru: B0 (Profil x Seviye). İlk HGB raporda "ilk model" olarak kalır.
- Üst sınır: S1, varsayılan q=0,90.
- Yük analizi ağ düzeyinde (toplam). `Location` yalnızca 14 mekân türü; tek tesis düzeyi veride yok. Kapasite kullanıcı parametresi olacak ve "gözlenen en yüksek yükün yüzdesi" gibi senaryo olarak sunulacak. Bu bir yöntem gösterimidir, tek tesis simülasyonu değildir.
- Kapasite referansı yalnızca eğitim+doğrulama dönemi yükünden hesaplanacak (test hariç).

## 10. Sınırlamalar

- Doğrulama yalnızca Haziran - Ağustos 2022 (üç kıvrım, tek yaz). Kış performansı ölçülmedi.
- B0 yapısı Ağustos tanısından sonra tasarlandı; Haziran ve Temmuz bağımsız destek verdi, nihai sınav Eylül test.
- Veride tatil bilgisi yok.
- U100/F50/F25 gerçek ölçüm değil.
- Kapasite verisi yok; kapasite senaryo parametresidir.

## 11. Önceki ifadelerden düzeltmeler

- "Oturumların %99,9'unda Duration bağlantı aralığıyla örtüşüyor" yanlıştı. Doğrusu: Duration, bağlantı aralığından 1 dakikadan az farkla uyuşan oturumlar yaklaşık %95; 5 dakikadan az farkla yaklaşık %97.
- İlk yük profili çalıştırması (69.966 oturum, ortalama 139,7 kWh/saat) hatalıydı (çift yönlü fark kuralı ve dakikaya yuvarlanmış zaman). Geçerli çalıştırma 72.039 oturumla.
- "Model zirveleri öğrenemiyor" yorumu eksikti; asıl sorun genel seviye yanlılığıdır.
- "XGBoost/LightGBM aynı sorunu taşır" bir tahmindi, test edilmedi.

## 12. Sıradaki işler

1. B0 tahminini ve üst sınırını saatlik yük senaryolarına çevirmek (oturum yayılım çekirdeği, yalnızca eğitim döneminden).
2. Kapasite parametresinin tanımı ve kapasite aşımı riski.
3. Kontrolsüz ve akıllı şarj senaryolarının karşılaştırılması (karşılanan talep, ertelenen enerji, aşım).
4. İsteğe bağlı: kısa model karşılaştırması (LightGBM, XGBoost, doğrusal) aynı kurallarla; tatil özelliği.
5. Streamlit karar destek arayüzü ve nihai rapor.
6. Eylül test: B0 ve üst sınır için bir kez.

## 13. Betikler (hepsi proje klasöründe)

sema_kontrol.py, dogrulama_tani.py, iyilestirme_denemeleri.py, referans_ve_artik_denemeleri.py, belirsizlik_denemeleri.py, dislama_ve_birim_tani.py, yaklasik_yuk_profili.py, durum_guncelle.py. Çıktı klasörleri: outputs_modeling\ (alt klasörler dogrulama_tani, iyilestirme_denemeleri, referans_ve_artik_denemeleri, belirsizlik_denemeleri) ve outputs_load_profile\.
"""

yol.write_text(baslik + icerik, encoding="utf-8")
print(f"DURUM.md oluşturuldu/güncellendi: {yol}")