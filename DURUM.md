# Proje Durum Dosyası

Son güncelleme: 14.11.2026 15:58


### 1. Proje ve değişmez kurallar

**Proje konusu:** Elektrikli Araç Şarj İstasyonlarında Saatlik Şarj Talebinin Makine Öğrenmesiyle Tahmini ve Operasyonel Karar Destek Sisteminin Geliştirilmesi.

**Projenin amacı:**
- Saatlik şarj oturumu başlangıç sayısını tahmin etmek.
- Oturum başlangıç saatine atfedilen enerji talebini tahmin etmek.
- U100, F50 ve F25 varsayımlarıyla yaklaşık saatlik yük profilini incelemek.
- B0 tahmin yöntemi ve S1 ihtiyatlı üst sınırıyla 24 saatlik tahmin sunmak.
- Operatörün talep yoğunluğunu ve olası yük değişimlerini inceleyebileceği bir karar destek arayüzü geliştirmek.

**Değişmez kurallar:**
- Orijinal `data\ChargingRecords.csv` asla değiştirilmez.
- `started_session_energy_kwh` gerçek şebeke yükü değildir; oturumun başladığı saate atfedilen toplam enerjidir.
- `ChargerType` 0/1 değerleri resmi veri sözlüğü olmadan AC/DC diye yorumlanmaz.
- Eylül 2022, ilk HGB modelinin test değerlendirmesinde kullanılmıştır. Bu nedenle tamamen bağımsız ve hiç görülmemiş test verisi olarak tanımlanmaz.
- B0 ve S1 yöntemleri Haziran–Ağustos 2022 rolling-origin doğrulaması üzerinden seçilmiştir.
- Yapılmamış iş yapılmış gibi gösterilmez; çıkarımlar "çıkarım" diye etiketlenir.
- U100, F50 ve F25 gerçek ölçülmüş yük dağılımları değil, varsayımsal duyarlılık senaryolarıdır.
- Gerçek şebeke/trafo kapasitesi ve gerçek zamanlı kontrol verisi bulunmadığından otomatik kapasite optimizasyonu veya şebeke müdahalesi iddia edilmez.

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

- **Ana tahmin yöntemi:** B0 (Profil × Seviye). İlk HGB modeli, başlangıç modelleme çalışması ve karşılaştırma amacıyla korunacaktır.
- **Belirsizlik üst sınırı:** S1, varsayılan q=0,90. Bu değer kesin güvenlik garantisi olarak yorumlanmayacaktır.
- **Tahmin hedefleri:** Saatlik oturum başlangıç sayısı, başlangıç saatine atfedilen enerji ve varsayımsal yaklaşık saatlik yük birbirinden ayrı çıktılar olarak gösterilecektir.
- **Yaklaşık yük senaryoları:** U100 varsayılan merkez senaryo; F50 ve F25 duyarlılık karşılaştırmalarıdır. Hiçbiri gerçek ölçülmüş şebeke yükü olarak sunulmayacaktır.
- **Veri kapsamı:** Yük analizi ağ düzeyindedir. `Location` yalnızca 14 mekân türü içerdiğinden tek tesis düzeyinde sonuç çıkarılmayacaktır.
- **Yoğun ve sakin saatler:** Yalnızca eğitim ve doğrulama döneminden elde edilen saatlik profiller kullanılarak belirlenecektir. Eylül 2022 bu amaçla eşik veya profil seçmek için kullanılmayacaktır.
- **Kapasite ve LLF:** Önceki kapasite analizi ve akıllı şarj simülasyonu deneyleri silinmeden arşiv çalışmaları olarak korunacaktır. Bunlar ana projenin doğrulanmış optimizasyon özellikleri olarak sunulmayacaktır.
- **Başarısız deneyler:** İkili kapasite aşımı uyarısı ve günlük aşım şiddeti sıralaması başarı ölçütlerini karşılamadığından doğrulanmış karar yetenekleri olarak gösterilmeyecektir.
- **Test dönemi:** Eylül 2022, ilk HGB modelinin değerlendirilmesinde kullanılmıştır. B0/S1 seçimi Haziran–Ağustos doğrulamasına dayanır; Eylül sonuçları bu geçmiş kullanım sınırlılığı açıkça belirtilerek raporlanacaktır.

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

## 12. Sıradaki işler ve tamamlanma planı

**Mevcut durum:** Veri hazırlığı, ilk modelleme, B0 karşılaştırmaları, S1 belirsizlik analizi, yaklaşık yük profili ve Aşama 1–2 Streamlit entegrasyonu tamamlandı. Aşama 1 ve Aşama 2 yedeklendi ve temel kontrollerden geçirildi.

### 1. Proje kapsamının güncellenmesi
- Ana proje amacı saatlik şarj talebi tahmini ve operasyonel karar desteği olarak sabitlenecek.
- Kapasite optimizasyonu ve LLF ana uygulama kapsamından çıkarılacak.
- Önceki simülasyonlar, deneyler ve çıktı dosyaları silinmeden arşiv niteliğinde korunacak.

### 2. Streamlit uygulamasının son düzenlemesi
- Saatlik oturum sayısı tahmini ayrı gösterilecek.
- Başlangıç saatine atfedilen enerji tahmini ayrı gösterilecek.
- Varsayımsal yaklaşık yük tahmini U100, F50 ve F25 senaryolarıyla ayrı gösterilecek.
- Başlangıç enerjisi gerçek şebeke yükü olarak adlandırılmayacak.
- U100 varsayılan senaryo, F50 ve F25 duyarlılık senaryoları olarak açıklanacak.
- B0 tahmini ve S1 üst sınır bandı gösterilecek.
- Yoğun ve sakin saatler yalnızca eğitim ve doğrulama verisinden hesaplanan profillere dayanarak sunulacak.
- Kapasite analizi gerekiyorsa varsayımsal senaryo olarak açıkça etiketlenecek.
- Doğrulanmamış otomatik uyarı ve günlük aşım şiddeti sıralaması ana özellik olarak sunulmayacak.

### 3. Uygulama kontrolleri
- app.py sözdizimi kontrolü yapılacak.
- Streamlit uygulamasının açılışı kontrol edilecek.
- Üç temel tahmin çıktısının birbirinden doğru ayrıldığı doğrulanacak.
- Grafikler, tablolar ve senaryo geçişleri kontrol edilecek.
- app_asama1_yedek.py ve app_asama2_yedek.py dosyaları korunacak.

### 4. Nihai değerlendirme ve raporlama
- Veri hazırlama ve dışlama kuralları belgelenecek.
- İlk HGB modelinin sonuçları ve tanısı açıklanacak.
- B0 yönteminin diğer yöntemlere göre seçilme gerekçesi sunulacak.
- S1 üst sınırının oluşturulması ve kapsama sonuçları açıklanacak.
- U100, F50 ve F25 varsayımlarının sınırlılıkları belirtilecek.
- Başarısız uyarı ve günlük şiddet deneyleri bilimsel dürüstlükle raporlanacak.
- Eylül 2022'nin ilk HGB değerlendirmesinde kullanılmış olduğu açıkça belirtilecek; B0/S1 sonuçları için bağımsız, hiç görülmemiş test iddiasında bulunulmayacak.

### 5. İsteğe bağlı çalışmalar
- LightGBM, XGBoost ve doğrusal modellerin karşılaştırılması zorunlu değildir.
- Yeni model araması mevcut uygulama düzenleme aşamasında başlatılmayacaktır.
- Ek deney yapılırsa veri sızıntısı ve test dönemi sınırlamaları korunacaktır.
   
## 13. Betikler (hepsi proje klasöründe)

sema_kontrol.py, dogrulama_tani.py, iyilestirme_denemeleri.py, referans_ve_artik_denemeleri.py, belirsizlik_denemeleri.py, dislama_ve_birim_tani.py, yaklasik_yuk_profili.py, durum_guncelle.py. Çıktı klasörleri: outputs_modeling\ (alt klasörler dogrulama_tani, iyilestirme_denemeleri, referans_ve_artik_denemeleri, belirsizlik_denemeleri) ve outputs_load_profile\.


## Güncelleme günlüğü

### 14.11.2026 16:05 - Deneme

Bu bir deneme notudur.

### 14.11.2026 16:12 - Yuk tahmini (B0 yuk serisi) sonuclari

B0, yuk serisine uygulandi (rolling-origin Haz-Tem-Agu 2022; profil ve kalibrasyon kivrimdan 2 gun once kesildi; seviye = origin anindaki baslangic enerjisi serisinin son 168 saat ortalamasi). Ortalama MAE kWh/saat: U100 35,07; F50 40,48; F25 45,39 (WAPE yuzde 19,4; 22,4; 25,2). Statik profil 51,29; 54,46; 58,09. Baslangic tahmini yuk sanilirsa 68,92; 56,15; 50,85. K1 ve K3 uc senaryoda PASS. q=0,90 ust sinir kapsama ortalamasi 0,891; 0,893; 0,898; Temmuz kivriminda 0,872-0,879 (nominalin biraz alti). Zirve saatlerinde tahmin gerceklesenin yaklasik yuzde 5-7 ustunde (kismen secim etkisi). B0 zirve saatlerini bulmada statik profilden iyi degil (recall 0,51-0,56 ve 0,55-0,59). Hatalar senaryo yukune gore, olculmus yuke gore degil. Test kullanilmadi. Karar: yuk tahmini = B0 yuk serisi ve S1 ust sinir. Siradaki is: kapasite parametresi ve akilli sarj simulasyonu.

### 14.11.2026 16:26 - Akilli sarj simulasyonu (Agustos 2022, kural tabanli denetleyici)

akilli_sarj_simulasyonu.py calistirildi; kontroller gecti (sonsuz kapasite = kontrolsuz; saatlik yuk CSV korelasyonu 0,992-0,994). 15 dk adim, 8016 degerlendirme oturumu, 142929,9 kWh. Kapasite = egitim donemi saatlik yuk yuzde 99 degerinin yuzde 60-90 kati (U100 icin 329,8 kW; F50 386,2; F25 429,7). Denetleyici: en az bosluklu oturum once (least laxity first), tahmin kullanmaz, her oturumun ayrilis zamanini ve enerjisini bilir (iyimser). U100 (bosluksuz, ana sonuc): karsilanamayan enerji yuzde 60 kapasitede 18,22 (esneklik yok) ve 11,70 (yuzde 50 esnek, 2 saat); yuzde 90 kapasitede 3,89 ve 0,28. Esnekligin etkisi esnek pay x ek bekleme carpimina bagli. F50/F25 bosluk varsayimdan geldigi icin sifira yakin karsilanamayan enerji mekanik sonuc; sadece iyimser duyarlilik analizi. Esnek kullanicilar rastgele secildi; kapasite tek ag toplami. Test kullanilmadi. Siradaki is: B0 yuk tahmini ve S1 ust sinirinin kapasite asimi saatlerini 24 saat onceden yakalama kalitesi (yakalama ve yanlis alarm), sonra Streamlit arayuzu.

### 14.11.2026 16:40 - Kapasite asimi uyari deneyi (uyari_denemeleri.py)

Olay: hedef saatte kontrolsuz saatlik ortalama yuk kapasiteyi asiyor mu; uyari 24 saat once; B0 ve S1 ile; kontroller gecti. Onceden yazilan olcutler U100 yuzde 70 ve yuzde 80 kapasitede: C2 (recall) PASS, C1 (F1 > en iyi karsilastirma) FAIL, C3 (precision >= 0,60) FAIL; genel FAIL. Yuzde 70: varsayilan kural (olasilik >= 0,25) precision 0,570 recall 0,846 F1 0,681, ayda 128 saat yanlis alarm; sabit pencere 17-22 F1 0,707; persistans-168 F1 0,591. B0 persistansi belirgin geciyor ama sabit pencereyi gecmiyor. Siralama kalitesi (AUC yaklasik 0,90) B0 ve statik profilde ayni; seviye olcekleme esik kalibrasyonunu duzeltiyor, siralamayi degil. Uyari kalitesi 1 ve 24 saat once arasinda ayni (B0 yakin zaman bilgisi kullanmiyor). Gun duzeyi uyari anlamsiz: yuzde 70 kapasitede gunlerin yuzde 98,9unda asim var. Sinirlama: sabit pencere EDA'dan secildi (tum yili kapsiyor), baseline hafif avantajli. Karar: ikili uyari dogrulanmis guclu yan olarak sunulmayacak; arayuzde asil cikti tahmini yuk, ust sinir bandi ve kapasite cizgisi; uyari esigi kaydirici olacak. Test kullanilmadi.

### 14.11.2026 19:15 - Günlük Aşım Enerjisi Şiddet Deneyi (A Aşaması)

- Amaç: İkili uyarının (aşım var/yok) gün düzeyinde aşırı doygun olması (%98,9 gün) nedeniyle, operatör için operasyonel planlama açısından kritik olan gün bazında aşım enerjisi (kapasiteyi aşan toplam kWh) şiddetini tahmin etme ve günleri şiddetine göre doğru sıralama yeteneğini ölçmek.
- Yöntem: Her hedef gün d için origin = (d-1) saat 23:00; forecast_horizon = 1..24 (00:00-23:00 kesintisiz saatlik tahmin). B0 = Profil x Seviye; S1 = saatlik ihtiyatlı üst sınır toplamı (ihtiyatlı skor; günlük %90 güven sınırı değildir). Karşılaştırma baseline'ları: Persistans-168 ve Profil-statik (aynı origin bilgi kümesiyle).
- Veri ve Bütünlük Kontrolleri: 92 hedef günün tamamı (Haziran 30, Temmuz 31, Ağustos 31 gün; 2.208 hedef saat) eksiksiz doğrulandı. Yük serisi zaman indeksi analiz aralığında kesintisiz 7.878 saat ile birebir eşleşti, tekrarlı/eksik/fazla saat ve sonlu olmayan değer bulunmadı. Eylül 2022 test verisi kesinlikle kullanılmadı.
- Kapasite Senaryoları: Kıvrım bazlı %99 persentil referansları (U100 için Haz 308,7 kW, Tem 314,0 kW, Ağu 329,9 kW). Eşik oranları %70, %80 ve %90.
- Sonuçlar (U100 Ana Senaryo, 92 gün):
  * %70 Kapasite: Havuzlanmış Spearman B0 = 0,420 (tanımlı) | Persistans-168 = 0,135 | Profil-statik = 0,345 | Pearson B0 = 0,448 | 3 kıvrım aritmetik ortalaması = 0,330.
    S1: PASS (0,420 > 0,345; B0 her iki baseline'ı geçti).
    S2: FAIL (Havuzlanmış 0,420 < 0,60; asgari kıvrım korelasyonu 0,243 < 0,45).
  * %80 Kapasite: Havuzlanmış Spearman B0 = 0,390 (tanımlı) | Persistans-168 = 0,146 | Profil-statik = 0,262 | Pearson B0 = 0,419 | 3 kıvrım aritmetik ortalaması = 0,331.
    S1: PASS (0,390 > 0,262; B0 her iki baseline'ı geçti).
    S2: FAIL (Havuzlanmış 0,390 < 0,60; asgari kıvrım korelasyonu 0,285 < 0,45).
  * %90 Kapasite: Havuzlanmış Spearman B0 = 0,366 | Persistans-168 = 0,145 | Profil-statik = NaN (Profil aşım üretmediği için varyans sıfır).
    S1: NOT EVALUABLE (Profil-statik Spearman tanımsız).
    S2: FAIL (0,366 < 0,60; min kıvrım 0,267 < 0,45).
    S3 (bilgi amaçlı): FAIL (0,366 < 0,50).
- GENEL KARAR: FAIL (U100 ana senaryoda %70 ve %80 kapasitelerde S1 sağlansa da S2 asgari operasyonel anlamlılık eşiği 0,60 ve kıvrım kararlılığı 0,45 sağlanamadı).
- Sınırlılıklar ve Çıkarımlar:
  * B0 modeli, hem Persistans-168'i hem de Profil-statik baseline'ını sıralama korelasyonunda geride bırakmıştır (S1 her iki kapasitede PASS).
  * Ancak elde edilen sıra korelasyonu (havuzlanmış r_s = 0,39 - 0,42; kıvrım ortalaması = 0,33), operatörün gün öncesi operasyonel planlamasında güvenebileceği önceden kilitlenmiş asgari eşiğin (0,60) belirgin biçimde altındadır.
  * Model genel seviye artışını yakalamakla birlikte, günlerin tepe aşım şiddetlerini sıralamada orta-düşük düzeyde kalmaktadır. Bu nedenle Streamlit arayüzünde "kesin gün sıralaması" veya otomatik şiddet alarmı birincil özellik yapılmamalı; arayüzün odağı saatlik talep tahmini, S1 üst sınır bandı ve esnek şarj simülasyonu parametreleri olmalıdır.
- Çıktı Dosyaları:
  * outputs_load_profile/siddet_deneyi/siddet_gun_detay.csv (828 satır; 3 senaryo x 3 kapasite x 92 gün)
  * outputs_load_profile/siddet_deneyi/siddet_korelasyonlar.csv (36 satır)
  * outputs_load_profile/siddet_deneyi/siddet_ozeti.txt

### 15.11.2026 11:57 - Aşama 2 Entegrasyonu

Aşama 2 kapasite analizi app.py içerisine entegre edildi. Aşama 1 sayısal değerleri korundu ve manuel kontroller yapıldı. Slider-sayısal giriş senkronizasyonu, kapasite aşım hesapları ve tarihsel simülasyon tablosu çalışıyor. Senaryo açıklamasında U100 sabit metin hatası, Güvenli Bölge ifadesi ve tarihsel simülasyon açıklamasında metin düzeltmeleri bekliyor.

### 15.11.2026 12:53 - Aşama 1 ve 2 Son Kontrolleri

15.11.2026: app.py güncel kodu py_compile kontrolünden hatasız geçti ve Streamlit üzerinde açıldı. Aşama 1 tasarımı ve hesaplamaları korundu. Aşama 2 senaryo adları ve açıklamaları düzeltildi. U100, F50 ve F25 senaryolarında kapasite referansları, B0/S1 aşım değerleri, grafikler ve tabloların senaryoya göre güncellendiği manuel olarak kontrol edildi. use_container_width için yalnızca kaldırılma uyarısı alındı; uygulamanın çalışmasına engel değil. app_asama1_yedek.py ve app_asama2_yedek.py yedekleri oluşturuldu. Bilimsel sınırlamalar korunacak; ikili aşım uyarısı ve günlük şiddet sıralaması doğrulanmış özellik olarak sunulmayacak.

### 15.11.2026 20:16 - app_yeni.py incelemesi

app_yeni.py (3 sekme: oturum baslangic, baslangic enerjisi, yaklasik yuk U100/F50/F25) kod okunarak incelendi; bellekte compile basarili; uygulama calistirilmadi. Yontem B0 ve S1 deneyleriyle tutarli; Eylul profil ve kalibrasyona alinmiyor. Sekme 3 U100 beklenen degerler: B0 toplam 4429,5 kWh, B0 tepe 316,5, S1 tepe 401,0, origin seviyesi 174,76 (henuz dogrulanmadi). Eksikler: hedef 4 (model karsilastirma ve performans) ve hedef 5 (yogun ve sakin saatler) arayuzde yok; U100 varsayilan merkez, F50 ve F25 duyarlilik etiketi eksik. app.py eski 3 asamali surum (Asama 3 LLF kapsam disi); arsivlenmesi onerildi. DURUM.md bayat kayitlari: baslik tarihi, Deneme notu, betik listesi, Asama 3 ve app_yeni.py kaydi.
