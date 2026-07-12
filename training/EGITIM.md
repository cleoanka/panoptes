# Panoptes Model Eğitim Rehberi

Bu rehber, kiralık bir GPU sunucusunda Panoptes'in kullanabildiği **her** modeli
sıfırdan veya fine-tune ile eğitmen için yazıldı. Her komut kopyala-yapıştır
çalışacak şekilde verildi; belirsiz olan her nokta (paket sürümüne bağlı API'ler
gibi) açıkça **[SÜRÜM-KONTROL]** etiketiyle işaretlendi — o adımda önce
`--help` çıktısını/dokümanı doğrula, sonra ilerle.

Buradaki paket sürümleri ve doğruluk rakamları 2026-07-12 tarihinde doğrulanmış
gerçeklerdir (`.build-notes/research-digest.md`). Rehberdeki tüm diğer süre ve
batch önerileri "başlangıç noktası" niteliğindedir, ölçüm değildir.

## İçindekiler

- [0. Genel bakış: hangi model neyi besliyor](#0-genel-bakış-hangi-model-neyi-besliyor)
- [1. Sunucu kurulumu](#1-sunucu-kurulumu)
- [2. Veri](#2-veri)
- [3. Araç dedektörü eğitimi](#3-araç-dedektörü-eğitimi)
- [4. Plaka dedektörü eğitimi](#4-plaka-dedektörü-eğitimi)
- [5. Plaka OCR fine-tune (fast-plate-ocr)](#5-plaka-ocr-fine-tune-fast-plate-ocr)
- [6. Renk sınıflandırıcı](#6-renk-sınıflandırıcı)
- [7. Marka/model sınıflandırıcı](#7-markamodel-sınıflandırıcı)
- [8. Sonuçları Panoptes'e bağlama](#8-sonuçları-panoptese-bağlama)
- [9. Sorun giderme](#9-sorun-giderme)

---

## 0. Genel bakış: hangi model neyi besliyor

Panoptes'te beş eğitilebilir model ailesi var. Her biri platformun farklı bir
modülünü besler:

| # | Model | Eğitim aracı | Beslediği modül / config alanı | Zorunlu mu? |
|---|-------|--------------|-------------------------------|-------------|
| 1 | Araç dedektörü (YOLO26 **veya** RF-DETR) | `training/train_detector.py` | `panoptes.detect` — `detector.backend` = `ultralytics` / `rfdetr` / `onnx` / `tensorrt` | Evet — platformun kalbi. Eğitmesen bile COCO ön-eğitimli ağırlıklar çalışır; fine-tune, van/emergency gibi COCO'da olmayan sınıfları ve kendi kamera açılarını kazandırır. |
| 2 | Plaka dedektörü (tek sınıf) | `training/train_plate_detector.py` | `panoptes.alpr` — plaka kutularını bulan aşama (`alpr.detector_model`) | Hayır — varsayılan open-image-models ağırlıkları hazır gelir; kendi modelin "temiz-oda" provenance için. |
| 3 | Plaka OCR (fast-plate-ocr fine-tune) | fast-plate-ocr'un kendi eğitim CLI'ı + `training/prepare/synth_plates.py` + `training/prepare/plate_crop.py` | `panoptes.alpr` — `alpr.ocr_model` | Hayır — global model TR plakalarda %97.8 plaka doğruluğuyla doğrulandı; fine-tune son yüzdeleri kovalamak için. |
| 4 | Renk sınıflandırıcı | `training/train_color.py` | `panoptes.attributes` — `attributes.color.method: model` + `model_path` | Hayır — bağımlılıksız HSV sezgiseli her zaman çalışır; model yöntemi gece/karma ışıkta daha isabetli. |
| 5 | Marka/model sınıflandırıcı | `training/train_makemodel.py` | `panoptes.attributes` — `attributes.makemodel.*` | Hayır — varsayılan kapalı; ticari-temiz veri bulmak bu alanda en zor iş (bkz. §7). |

### 0.1 Lisans katmanları — satılabilir ürün için ne anlama geliyor

Panoptes kodu Apache-2.0. Ama **eğittiğin ağırlıkların lisansı, eğitimde
kullandığın framework'ün ve verinin lisansından türer.** İki dedektör yolu var
ve fark ticari:

| | YOLO26 yolu (ultralytics) | RF-DETR yolu (Apache-temiz) |
|---|---|---|
| Paket | `ultralytics==8.4.92` | `rfdetr==1.8.3` |
| Kod lisansı | **AGPL-3.0** | Apache-2.0 |
| Ağırlık lisansı | **AGPL-3.0 — fine-tune'lar dahil.** Senin eğittiğin `best.pt` de AGPL olur. | Apache-2.0 — **yalnızca Nano/Small/Medium/Large.** XL/2XL katmanları PML-1.0'dır; indirme, fine-tune etme, paketleme. |
| COCO doğruluğu (doğrulanmış) | n 40.9 / s 48.6 / m 53.1 / l 55.0 / x 57.5 mAP | Nano 48.4 … Large 56.5 AP |
| T4 hız (doğrulanmış) | 1.7–11.8 ms (TensorRT) | 2.3–6.8 ms (FP16) |
| Ticari sonuç | Kapalı-kaynak/satılan üründe kullanmak için **Ultralytics Enterprise License satın alman gerekir**; almazsan AGPL'in ağ-servisi maddesi dahil tüm yükümlülükleri (kaynak açma) geçerlidir. | Satılabilir üründe kısıtsız. Panoptes'in "license-clean default" dedektörü budur. |

Pratik strateji:

- **Ar-ge / iç değerlendirme:** YOLO26 ile eğit — takım olgun, dokümantasyon
  zengin, hiperparametre otomasyonu iyi. AGPL, dahili kullanımda sorun değildir.
- **Müşteriye giden build:** ya RF-DETR ağırlığı eğit (Apache), ya Ultralytics
  Enterprise License satın al ve sözleşme numarasını
  `deploy/weights_manifest.yaml`'a işle. Üçüncü tam-Apache yedek: D-FINE
  (`transformers` üzerinden `DFineForObjectDetection`,
  `ustc-community/dfine_l_coco`) — ama `obj2coco` varyantlarını kullanma,
  Objects365 şartları akademik-sadece.
- ALPR yığını (fast-plate-ocr + open-image-models) MIT'dir; yalnızca
  open-image-models'ın hazır plaka-dedektör ağırlıklarının YOLOv9 (GPL upstream)
  soyu vardır — manifest'te `ship_ok: review` olarak işaretli. §4'teki kendi
  plaka dedektörün bu riski tamamen kaldırır.
- Veri tarafındaki katmanlar §2.1'de.

---

## 1. Sunucu kurulumu

### 1.1 Donanım önerisi

| Bileşen | Öneri | Neden |
|---|---|---|
| GPU | **L40S 48GB** (ilk tercih) veya RTX 4090 24GB / RTX 5090 32GB | 100k görüntülük dedektör eğitimi 24GB'a sığar; 48GB batch/imgsz özgürlüğü verir |
| vCPU | ≥ 16 | Ultralytics dataloader'ı augment'i CPU'da yapar; az çekirdek = aç GPU |
| RAM | ≥ 64 GB | `cache="disk"` bile sayfa önbelleğinden beslenir |
| Disk | NVMe, ≥ 500 GB | COCO tek başına ~20 GB ham + dönüştürülmüş kopyalar |
| OS | Ubuntu 24.04 LTS | Bu rehberdeki tüm komutlar buna göre |

Kiralama notu: Vast/Lambda/RunPod tarzı sağlayıcılarda sürücü genelde hazır
kurulu gelir — önce `nvidia-smi` dene; çalışıyorsa §1.2'nin sürücü adımını atla.

RTX 5090 (Blackwell, `sm_120`) kiralarsan: PyPI'daki varsayılan torch wheel'i
bu mimariyi desteklemeyebilir; §1.3'teki cu128 index satırını kullan.

### 1.2 Taban sistem + NVIDIA sürücüsü

```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y build-essential git curl unzip tmux htop nvtop ubuntu-drivers-common

# Sürücü yoksa (nvidia-smi hata veriyorsa):
sudo ubuntu-drivers install
sudo reboot
```

Yeniden bağlanınca doğrula:

```bash
nvidia-smi
# Sağ üstte "CUDA Version: 12.x" (veya üstü) görmelisin.
```

Ayrı bir CUDA Toolkit kurulumu **gerekmez**: torch/ultralytics wheel'leri CUDA
runtime'ını kendi içinde taşır; yalnızca sürücü yeterli.

### 1.3 Python 3.12 + uv + repo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12

git clone https://github.com/cleoanka/panoptes.git ~/panoptes
cd ~/panoptes
uv venv --python 3.12
source .venv/bin/activate

# YOLO26 yolu + Roboflow indirici (ultralytics AGPL — bkz. §0.1):
uv pip install -e '.[train]'

# Apache yolu (RF-DETR):
uv pip install -e '.[rfdetr]'

# Renk / marka-model sınıflandırıcıları için:
uv pip install torch torchvision
# RTX 5090 / Blackwell ise bunun yerine:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Export doğrulaması ve ONNX değerlendirmesi için:
uv pip install onnxruntime-gpu==1.27.0
```

`onnxruntime-gpu==1.27.0` notu: bu sürüm CUDA 12 ile çalışır ama ORT, CUDA 12
desteğini "deprecated" ilan etti — CUDA 13 geçişi ufukta. Sunucun CUDA 13
sürücüsüyle geldiyse ORT'un CUDA 13 wheel'ini kur ([SÜRÜM-KONTROL]).

Kurulum sağlaması:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import ultralytics; print(ultralytics.__version__)"   # 8.4.92 bekleniyor
```

### 1.4 Veri diski ve oturum düzeni

```bash
sudo mkdir -p /data/raw /data/datasets
sudo chown -R "$USER:$USER" /data
```

Uzun eğitimleri **mutlaka** tmux içinde başlat — SSH koparsa eğitim ölmesin:

```bash
tmux new -s egitim
# ... eğitim komutu ...
# Ayrıl: Ctrl-b d   Geri dön: tmux attach -t egitim
```

---

## 2. Veri

### 2.1 Lisans-katmanlı veri seti tablosu

**Katman kuralı:** TİCARİ-TEMİZ setlerle eğitilen ağırlıklar müşteriye
gidebilir. ARAŞTIRMA-SADECE setler yalnızca *benchmark/karşılaştırma* içindir —
onlarla eğitilmiş bir ağırlığı asla satılan build'e koyma; manifest'e girecekse
`ship_ok: false` yaz.

#### TİCARİ-TEMİZ

| Set | Lisans | Boyut | Erişim | Not |
|---|---|---|---|---|
| COCO 2017 | Anotasyonlar CC BY 4.0 | 118k train + 5k val | Doğrudan indirme (§2.3) | 80-sınıf düzeninde araç id'leri: person=0, bicycle=1, car=2, motorcycle=3, bus=5, train=6, truck=7. `van` ve `emergency` YOK. |
| Roboflow `vehicle-mscoco/vehicles-coco` | CC BY 4.0 | ~19k | `roboflow_pull.py --name vehicles-coco` | COCO-türevi araç sınıfları; genel dedektör tabanı |
| Roboflow `roboflow-universe-projects/license-plate-recognition-rxg4e` | CC BY 4.0 | 10,125 | `roboflow_pull.py --name plates-universe` | Tek sınıf plaka; plaka dedektörü tabanı |
| Roboflow `kemalkilicaslan-gzpvq/license-plates-of-vehicles-in-turkey-s3tbj` | CC BY 4.0 | ~3.5k | `roboflow_pull.py --name tr-plates-kemalkilicaslan` | TR plaka |
| Roboflow `dsds-hjsno/turkish-number-plates-bvgm0` | CC BY 4.0 | 2,246 | `roboflow_pull.py --name tr-plates-dsds` | TR plaka |
| Roboflow `tr-plaka-recognition/tr-plaka-dataset` | CC BY 4.0 | ~1.4k | `roboflow_pull.py --name tr-plaka-dataset` | TR plaka |
| CCPD 2019/2020 | MIT | ~300k | `github.com/detectRecog/CCPD` | **Çin** plakaları — TR OCR'ına alfabe uymaz; yalnızca plaka-*dedektörü* verisi olarak işe yarar |
| AI City Challenge | İstek-bazlı | değişken | Resmi site, başvuru formu | Trafik kamerası açıları; erişim onayı bekle |
| Sentetik TR plaka | Kendi kodumuz (Apache) | sınırsız | `training/prepare/synth_plates.py` | TR OCR için ticari-temiz ölçekleme yolu (§2.5) |
| Kendi kameralarından toplanan veri | Senin | sınırsız | Panoptes snapshot'ları | KVKK dayanağını belgele (§9.4) |

CC BY 4.0 yükümlülüğü: atıf ver — ürün dokümanına/NOTICE dosyasına set adı +
kaynak URL yazmak yeterlidir. Roboflow Universe yazarları lisansı sonradan
değiştirebilir; **ağırlık yayınlamadan önce projenin lisans sayfasını yeniden
kontrol et** (indirme scripti de bu uyarıyı basar).

#### ARAŞTIRMA-SADECE (benchmark için; satılabilir ağırlık eğitme!)

| Set | Durum | Not |
|---|---|---|
| BDD100K | Berkeley OTL — ticari kullanım ayrı lisans ister | Zengin sürüş sahneleri; yalnız değerlendirme |
| nuImages | Araştırma lisansı | |
| VisDrone | Araştırma | Drone açısı — Panoptes'in tipik sabit-kamera senaryosuna zaten uzak |
| UA-DETRAC | Mirror'lar araştırma-sadece | **Tuzak:** ignore-region maskeleri var; maskesiz değerlendirirsen mAP'in haksız düşer |
| CompCars | İmzalı kullanım sözleşmesi (agreement) gerekir | Marka/model; sözleşme ticari kullanımı kapsamaz |
| Stanford Cars | **Resmi linkler ÖLÜ** (Stanford sayfası kalktı) | Kaggle mirror'ları var ama lisans tanımsız — satılabilir ürün için kullanma (§7) |
| VMMRdb | Araştırma | Marka/model |
| VeRi-776 / VehicleID | E-posta ile akademik erişim | Re-ID setleri |
| UFPR-ALPR / RodoSol | Üniversite e-postası şart | Brezilya plakaları; yalnız OCR benchmark |
| VCoR (renk) | Kaggle, araştırma varsayımıyla ele al | Renk sınıflandırıcı benchmark'ı (§6) |

### 2.2 Roboflow'dan indirme

Ücretsiz hesap aç, API anahtarını al (app.roboflow.com → Settings → API), sonra:

```bash
export ROBOFLOW_API_KEY=BURAYA_ANAHTARIN

cd ~/panoptes
python training/prepare/roboflow_pull.py --list                       # kayıtlı setler
python training/prepare/roboflow_pull.py --name vehicles-coco --out /data/raw
python training/prepare/roboflow_pull.py --all --out /data/raw        # hepsi (YOLO formatı)

# RF-DETR eğitimi COCO formatı ister:
python training/prepare/roboflow_pull.py --name vehicles-coco --out /data/raw/coco-fmt --format coco
```

`--format yolov11` (varsayılan) düzeni YOLO26 için de birebir uyar; Roboflow'un
ayrı bir "yolo26" export'u olup olmadığına bakma gereği yok.

Tekrarlanabilirlik için sürüm sabitle: `--version N` (yazmadığında en son
yayınlanmış sürüm iner ve script hangisini indirdiğini basar — not al).

### 2.3 COCO indirme + YOLO formatına dönüştürme

```bash
cd /data/raw && mkdir -p coco && cd coco
curl -LO http://images.cocodataset.org/zips/train2017.zip
curl -LO http://images.cocodataset.org/zips/val2017.zip
curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q train2017.zip && unzip -q val2017.zip && unzip -q annotations_trainval2017.zip
rm -f train2017.zip val2017.zip annotations_trainval2017.zip
```

Dönüştürme + kanonik sınıflara remap (tek komut; `ultralytics.data.converter.convert_coco`
üzerine ince bir sarmalayıcıdır, ardından etiketleri yerinde 8-sınıf düzenine çevirir):

```bash
cd ~/panoptes
python training/prepare/coco_to_yolo.py \
  --coco-labels-dir /data/raw/coco/annotations \
  --save-dir /data/datasets/vehicles-conv
```

Anotasyon dizinindeki `captions_*.json` / `person_keypoints_*.json`
dosyalarını script otomatik dışarıda bırakır (yalnızca `instances_*.json`
dönüştürücüye gider) — ayıklaman gerekmez. Tekrar koşularda script mevcut
`--save-dir`'i görürse **durur**: yarıda kalan bir dönüşümün üstüne koşmak
etiketleri ikinci kez remap edip sınıf id'lerini sessizce karıştırırdı.
Temiz dönüşüm için dizini sil ya da `--force` ver (önce siler).

`convert_coco` yalnızca etiket üretir; görüntüleri kendin bağlarsın. Nihai
düzeni `training/configs/vehicles.yaml`'ın beklediği hâle getir:

```bash
mkdir -p /data/datasets/vehicles/images /data/datasets/vehicles/labels
ln -s /data/raw/coco/train2017 /data/datasets/vehicles/images/train
ln -s /data/raw/coco/val2017   /data/datasets/vehicles/images/val
mv /data/datasets/vehicles-conv/labels/train2017 /data/datasets/vehicles/labels/train
mv /data/datasets/vehicles-conv/labels/val2017   /data/datasets/vehicles/labels/val
```

Not: dönüştürücünün yazdığı alt-klasör adları anotasyon dosya adından türer
(`instances_train2017.json` → `labels/train2017`); farklıysa `mv` satırlarını
ona göre uyarla.

Özel bir kaynaktan farklı id düzeniyle gelen YOLO etiketleri için remap'i JSON
ile ezebilirsin:

```bash
cat > /tmp/remap.json <<'EOF'
{"0": 0, "1": 2, "2": 3}
EOF
python training/prepare/coco_to_yolo.py --coco-labels-dir ... --save-dir ... --remap-json /tmp/remap.json
```

### 2.4 Kanonik sınıf sözleşmesi (`training/configs/vehicles.yaml`)

```yaml
names:
  0: car
  1: van          # COCO'da YOK
  2: bus
  3: truck
  4: motorcycle
  5: bicycle
  6: emergency    # COCO'da YOK (ambulans+polis+itfaiye tek sınıf)
  7: person
```

Bu id sıralaması **sözleşmedir**: `coco_to_yolo.py` bu sıraya remap eder ve
`panoptes.detect.classmap.map_label` bu *isimleri* tanır (Türkçe eşdeğerler
dahil: `kamyon`, `otobus`, `ambulans`...). Kendi verini etiketlerken bu
isimleri kullanırsan eğitilmiş model platforma **kod değişikliği olmadan**
takılır. Anotasyona başladıktan sonra id'leri asla yeniden sıralama.

Bu yaml'daki sekiz kanonik ismin tamamı — düz `emergency` dahil — classmap
tarafından tanınır; `ambulance`/`police`/`fire_truck` gibi özgül etiketler de
ayrıca EMERGENCY'ye eşlenir. Kod tarafında hiçbir değişiklik gerekmez.

`van` ve `emergency` boşluğunu doldurma yolları:

1. Roboflow Universe'te CC BY 4.0 lisanslı van/ambulance setleri ara ve
   `roboflow_pull.py`'daki `REGISTRY` sözlüğüne ekle (lisansı sayfadan doğrula);
2. kendi kameralarından topla ve etiketle (CVAT/Label Studio, YOLO export) —
   KVKK notu §9.4;
3. bu iki sınıf boş kalırsa da eğitim çalışır — model o sınıfları hiç tahmin
   etmez, platform geri kalanıyla normal işler.

### 2.5 TR plaka verisi + sentetik plaka üretici

Gerçek TR setleri (§2.2'deki üç `tr-*` kaydı, toplam ~7k görüntü) plaka
*dedektörü* için yeterli ama OCR fine-tune için azdır. Ticari-temiz ölçekleme
yolu sentetik üretimdir:

```bash
# 20.000 augment'li OCR crop'u + labels.csv (image_path,plate_text):
python training/prepare/synth_plates.py --out /data/datasets/tr-plates-synth --count 20000 --seed 42

# Önce gözle kontrol etmek istersen (augment'siz temiz render):
python training/prepare/synth_plates.py --out /tmp/plate-preview --count 12 --clean
```

Üretici TR plaka gramerini birebir uygular — il kodu 01–81, yasal harf
alfabesi (Q, W, X ve noktalı Türkçe harfler yok), harf sayısına göre doğru
basamak sayısı — ve her etiket `panoptes.alpr.validate.validate(text, "TR")`
doğrulamasından geçer (birim testi bunu 100/100 şart koşar). Augment zinciri
perspektif, pozlama, blur/motion-blur, küçültme, sensör gürültüsü ve JPEG
artefaktı basamaklarından oluşur.

Font dürüstlüğü: gerçek TR plakaları DIN 1451 türevi lisanslı bir yazıyla
basılır; üretici bağımlılıksız kalmak için OpenCV Hershey Duplex kullanır.
64×128'e küçülen OCR girdisinde fark büyük ölçüde kaybolur; yine de en iyi
sonuç için sentetiği her zaman gerçek crop'larla karıştır (§5.1).

---

## 3. Araç dedektörü eğitimi

İki yol da aynı script'ten çıkar: `training/train_detector.py`. Çıktılar
`training/runs/<isim>/` altına düşer.

### 3.1 YOLO26 yolu (ultralytics 8.4.92 — AGPL, bkz. §0.1)

```bash
cd ~/panoptes && source .venv/bin/activate
tmux new -s egitim

python training/train_detector.py \
  --data training/configs/vehicles.yaml \
  --model yolo26s.pt \
  --epochs 100 --imgsz 640 --batch -1 \
  --workers 16 --patience 20 \
  --name vehicles-y26s
```

Script'in ultralytics'e geçirdiği doğrulanmış eğitim parametreleri:
`batch=-1` (GPU'ya göre otomatik batch), `cache="disk"`, `cos_lr=True`,
`patience=20`; optimizer 8.4.x'te varsayılan-otomatik MuSGD'dir, elleme.
YOLO26 uçtan-uca NMS'sizdir; `iou` parametresi bu ailede anlamsızdır.
8.4.x'te `half`/`int8` argümanları deprecated olup yerlerini birleşik
`quantize` aldı — script hiçbirini geçirmez, sen de ekleme.

**Model boyu seçimi** (COCO mAP değerleri doğrulanmış):

| Model | mAP | Ne zaman |
|---|---|---|
| yolo26n | 40.9 | Edge/CPU hedefi, çok akış tek GPU |
| yolo26s | 48.6 | **Varsayılan başlangıç** — hız/doğruluk dengesi |
| yolo26m | 53.1 | Tek GPU'da hâlâ gerçek-zamanlı, belirgin doğruluk sıçraması |
| yolo26l/x | 55.0 / 57.5 | Batch-video işleme, doğruluk kritikse |

**Süre tahmini** (doğrulanmış aralık): yolo26m @ 640, 100k görüntü, 100 epoch →
4090/L40S/A100 sınıfında **~6–14 saat**. `s` modeli ve/veya daha küçük veriyle
kabaca orantılı düşer.

**Küçük veri (<~1k görüntü) fine-tune reçetesi** — doğrulanmış öneri seti
(`mosaic=0.5, mixup=0, lr0=0.001, freeze=10` + 50 epoch) tek bayrakla gelir:

```bash
python training/train_detector.py \
  --data training/configs/vehicles.yaml \
  --model yolo26s.pt --epochs 50 --finetune --name vehicles-ft
```

### 3.2 Değerlendirme (mAP + sınıf-bazlı)

```bash
python - <<'PY'
from ultralytics import YOLO
m = YOLO("training/runs/vehicles-y26s/weights/best.pt")
metrics = m.val(data="training/configs/vehicles.yaml")
print(f"mAP50-95: {metrics.box.map:.4f}   mAP50: {metrics.box.map50:.4f}")
for i, ap in enumerate(metrics.box.maps):
    print(f"  class {i}: mAP50-95 = {ap:.4f}")
PY
```

Sınıf-bazlı satırlara mutlaka bak: `van`/`emergency` verisi eklemediysen o
id'ler 0 görünür (normal); `motorcycle`/`bicycle` düşükse genelde küçük-nesne
sorunudur → `--imgsz 960` ile bir deneme koşusu yap.

### 3.3 RF-DETR yolu (Apache-temiz)

Veri **COCO formatında bir dizin** olmalı (Roboflow `--format coco` export'u:
`train/valid/test` alt dizinleri, her birinde `_annotations.coco.json`):

```bash
python training/train_detector.py \
  --data /data/raw/coco-fmt/vehicles-coco \
  --model rfdetr-small \
  --epochs 50 --batch 4 --grad-accum 4 --lr 1e-4 \
  --name vehicles-rfdetr
```

**[SÜRÜM-KONTROL]** Dürüst uyarı: rfdetr'in eğitim API'ı ultralytics kadar
oturmuş değildir. Script, README konvansiyonundaki kwarg'ları kullanır
(`dataset_dir`, `epochs`, `batch_size`, `grad_accum_steps`, `lr`,
`output_dir`). Kurulu `rfdetr==1.8.3` bir argümanı reddederse önce şunu çalıştır
ve script'teki çağrıyı çıktıya göre uyarla:

```bash
python -c "from rfdetr import RFDETRSmall; help(RFDETRSmall.train)"
```

Bellek notu: DETR aileleri YOLO'dan daha çok VRAM ister — 24GB kartta
`--batch 4 --grad-accum 4` (etkin batch 16) iyi bir başlangıçtır; OOM'da önce
`--batch 2 --grad-accum 8`.

XL/2XL katmanlarını script zaten reddeder (PML-1.0 — ticari yasak).

### 3.4 Export: ONNX / TensorRT

```bash
# ONNX (taşınabilir; Panoptes'in onnx backend'i doğrudan yer):
python - <<'PY'
from ultralytics import YOLO
out = YOLO("training/runs/vehicles-y26s/weights/best.pt").export(format="onnx", imgsz=640)
print(out)
PY

# Aynısı platform CLI'ından:
panoptes export --model training/runs/vehicles-y26s/weights/best.pt --format onnx --imgsz 640

# TensorRT engine — DEPLOYMENT GPU'SUNDA üret (engine dosyası GPU mimarisine özgüdür;
# L40S'te ürettiğin engine T4'te açılmaz):
panoptes export --model training/runs/vehicles-y26s/weights/best.pt --format engine --imgsz 640
```

Ultralytics'in ONNX export'u sınıf isimlerini modelin metadata'sına gömer;
Panoptes'in onnx backend'i bunları otomatik okur ve `classmap` kanonik
sınıflara çevirir — isim dosyası taşımana gerek yok. Metadata'sız üçüncü-parti
ONNX için `detector.extra.names` ile elle verirsin (§8.1).

RF-DETR fine-tune'u platforma `detector.extra.rfdetr_kwargs`
(`pretrain_weights`) ile takılır — örnek config ve sınıf-ismi uyarısı için
§8.1'e bak.

### 3.5 Eğitilen modeli Panoptes'te ilk test

```bash
mkdir -p ~/panoptes/weights
cp training/runs/vehicles-y26s/weights/best.onnx ~/panoptes/weights/vehicles.onnx

cat > /tmp/test-detector.yaml <<'EOF'
detector:
  backend: onnx
  model: ./weights/vehicles.onnx
  imgsz: 640
  conf: 0.3
streams:
  - id: test
    source: /data/raw/ornek-video.mp4
EOF

panoptes validate-config -c /tmp/test-detector.yaml
panoptes process /data/raw/ornek-video.mp4 --config /tmp/test-detector.yaml
```

---

## 4. Plaka dedektörü eğitimi

Tek sınıf (`license_plate`), küçük nesne. Nano-boy modeller yeterlidir.

### 4.1 Veri hazırlama

```bash
# Üç TR seti + büyük genel set, YOLO formatında:
python training/prepare/roboflow_pull.py --name plates-universe          --out /data/raw
python training/prepare/roboflow_pull.py --name tr-plates-kemalkilicaslan --out /data/raw
python training/prepare/roboflow_pull.py --name tr-plates-dsds           --out /data/raw
python training/prepare/roboflow_pull.py --name tr-plaka-dataset         --out /data/raw
```

Setleri `training/configs/plates.yaml`'ın beklediği tek düzende birleştir
(hepsi tek sınıf olduğundan id çakışması yoktur; dosya adı çakışmasına karşı
kopyalarken set adını öne koy):

```bash
mkdir -p /data/datasets/plates/images/{train,val} /data/datasets/plates/labels/{train,val}
# Dizin adları roboflow_pull.py'nin kayıt anahtarlarıdır (indirme hedefi <out>/<anahtar>):
for set in plates-universe tr-plates-kemalkilicaslan tr-plates-dsds tr-plaka-dataset; do
  for split in train valid; do
    dst=$([ "$split" = "valid" ] && echo val || echo train)
    src=$(find /data/raw -maxdepth 3 -type d -path "*${set}*/${split}" | head -1)
    [ -z "$src" ] && continue
    for f in "$src"/images/*; do cp "$f" "/data/datasets/plates/images/$dst/${set}_$(basename "$f")"; done
    for f in "$src"/labels/*; do cp "$f" "/data/datasets/plates/labels/$dst/${set}_$(basename "$f")"; done
  done
done
find /data/datasets/plates/images -type f | wc -l   # beklenen: ~17k dosya
```

(Roboflow export'larının dizin adları sürüme göre değişebilir; `find` satırı
boş dönerse `ls /data/raw` ile gerçek adlara bak ve `set` listesini düzelt.)

### 4.2 Eğitim

```bash
# YOLO26n yolu (fine-tune preset'i bu script'te VARSAYILAN AÇIK — plaka setleri küçük):
python training/train_plate_detector.py \
  --data training/configs/plates.yaml \
  --model yolo26n.pt --epochs 80 --name plates

# Büyük birleşik sette preset'i kapat:
python training/train_plate_detector.py --data training/configs/plates.yaml \
  --model yolo26n.pt --epochs 100 --no-finetune --name plates-full

# Temiz-oda (Apache) yolu — COCO formatlı dizinle:
python training/prepare/roboflow_pull.py --name plates-universe --out /data/raw/coco-fmt --format coco
python training/train_plate_detector.py \
  --data /data/raw/coco-fmt/plates-universe \
  --model rfdetr-nano --epochs 60 --batch 4 --name plates-rfdetr
```

Süre: tek sınıf ~17k görüntüde yolo26n genellikle 1–3 saat mertebesindedir
(başlangıç-noktası tahmini, ölçüm değil).

Neden eğitmeye değer: varsayılan open-image-models ağırlıkları MIT paketle
gelse de YOLOv9 (GPL upstream) soyludur — manifest'te `ship_ok: review`.
Kendi eğittiğin dedektör provenance'ı tamamen temizler. Platforma takma
seçenekleri §8.2'de.

---

## 5. Plaka OCR fine-tune (fast-plate-ocr)

Önce dürüst bir soru: **gerçekten gerekli mi?** Doğrulanmış gerçek: hazır
`cct-s-v2-global-model` TR plakalarında %97.8 plaka / %99.63 karakter
doğruluğu verir ve Panoptes'in üstünde ayrıca yapı-farkında düzeltme
(`panoptes/alpr/validate.py`) + track-bazlı oylama katmanı vardır. Fine-tune,
kendi kameralarının özgün koşullarında (açı, gece, IR) son yüzdeleri kovalamak
içindir.

### 5.1 Eğitim verisi: sentetik + gerçek karışımı

```bash
# 1) Sentetik taban (§2.5):
python training/prepare/synth_plates.py --out /data/datasets/tr-plates-synth --count 20000 --seed 42

# 2) Gerçek crop'lar — YOLO-formatlı plaka setinden kes, metinleri global modelle ön-doldur:
uv pip install 'fast-plate-ocr[onnx]'
python training/prepare/plate_crop.py \
  --dataset /data/datasets/plates \
  --out /data/datasets/tr-plates-real \
  --margin 0.08 --ocr-prefill
```

`--dataset` iki düzeni de kabul eder: §4.1'de birleştirdiğin
`images/{train,val}` + `labels/{train,val}` ağacı (yukarıdaki komut) veya
ham Roboflow export'u (`<split>/images` + `<split>/labels` — tek bir seti
kesmek istersen `--dataset /data/raw/tr-plates-dsds` gibi doğrudan ver).

`--ocr-prefill` bootstrap etiketlemedir: her plakayı sıfırdan yazmak yerine
modelin okumasını **düzeltirsin**. `labels.csv`'yi bir tabloda aç, `plate_text`
sütununu görüntüyle karşılaştırarak elden geçir — yanlış etiketle fine-tune,
modeli hazır hâlinden geriye götürür.

İki setin `labels.csv` düzeni aynıdır (`image_path,plate_text`); birleştirip
%90/%10 train/val böl.

### 5.2 fast-plate-ocr eğitim CLI'ı

**[SÜRÜM-KONTROL]** Eğitim, fast-plate-ocr projesinin kendi araç zinciriyle
yapılır; bu depo yalnızca veriyi hazırlar. Burada verilen akış paketin genel
konvansiyonudur — kesin bayrak adlarını kurulumundan doğrula; inference API'ı
dışındaki kısımlar bizim doğrulanmış gerçekler listemizde yer almıyor:

```bash
uv pip install 'fast-plate-ocr[train]'
fast-plate-ocr --help          # gerçek komut/bayrak adlarını buradan al
```

Konsol komutu paket adı gibi **tirelidir** (`fast-plate-ocr` — 1.1.0
wheel'inin tek console entry point'i bu; `python -m` karşılığı yok).
Wheel'de görünen alt komutlar: `train`, `valid`, `export`,
`validate-dataset`, `visualize-augmentation` ([SÜRÜM-KONTROL] — kurulumundan
`--help` ile doğrula).

Doğrulaman gereken üç nokta:

1. **Anotasyon formatı** — bizim ürettiğimiz `labels.csv`
   (`image_path,plate_text` başlıklı, görüntüye göreli yol) paketin beklediği
   CSV'ye ya birebir uyar ya da tek `awk` satırıyla çevrilir; paketin docs'unda
   örnek CSV ile karşılaştır.
2. **Plaka konfigi** — TR için maksimum slot sayısı 8'dir
   (2 rakam + 1–3 harf + 2–5 rakam kombinasyonlarının en uzunu 8 karakter),
   alfabe `A-Z0-9` + pad karakteri (`_`).
3. **Checkpoint'ten devam** — mümkünse sıfırdan değil
   `cct-s-v2-global-model`'den fine-tune başlat (destek bayrağını docs'tan bul);
   65 ülkelik ön-eğitim, sentetik/gerçek karışımının azlığını telafi eder.

Eğitim ONNX çıktısı verir; Panoptes'e takma seçenekleri §8.2'de.

### 5.3 Fine-tune'u değerlendirme

Elde-düzeltilmiş gerçek crop'lardan ~500 taneyi eğitime hiç sokma; şu iki metriği
hem global modelle hem fine-tune'unla karşılaştır: plaka-tam-doğruluk
(tüm karakterler doğru) ve karakter-doğruluk. Fine-tune, global modeli
**geçemiyorsa üretime koyma** — %97.8'lik taban zaten güçlü.

---

## 6. Renk sınıflandırıcı

Varsayılan HSV sezgiseli (`attributes.color.method: heuristic`) bağımlılıksızdır
ve gündüz sahnelerinde yeterlidir. Model yöntemi gece/sodyum-buharı aydınlatma
altında kazanır.

### 6.1 Veri

On kanonik renk klasör adıdır (platform sözlüğüyle birebir):
`white black gray silver red blue green yellow orange brown`.

```
/data/datasets/color/
  train/white/*.jpg  train/black/*.jpg ... train/brown/*.jpg
  val/white/*.jpg    ...
```

Kaynak seçenekleri:

- **Kendi verin (önerilen, ticari-temiz):** Panoptes'i mevcut COCO ağırlıklarıyla
  birkaç gün koştur, `TRACK_FINISHED` snapshot'larındaki araç crop'larını
  renklerine göre klasörle. Kendi kameralarının ışık koşullarını birebir öğrenir.
- **VCoR (Kaggle):** araştırma katmanında say — benchmark için kullan,
  satılacak ağırlığı kendi verinle eğit.

Sınıf başına birkaç yüz görüntü yeterli başlangıçtır; `silver`/`gray` sınırı
en çok karışan bölgedir, oraya örnek yığ.

### 6.2 Eğitim + ONNX export

```bash
python training/train_color.py \
  --data /data/datasets/color \
  --arch resnet18 --epochs 40 --batch 64 --imgsz 224 \
  --out training/runs/color/color.onnx
```

Script cosine LR + erken durdurma (patience 8) uygular, en iyi epoch'u ONNX'e
verir (`opset 17`, statik `[1,3,224,224]`), etiketleri `color.txt` yan
dosyasına yazar ve onnxruntime kuruluysa torch↔onnx çıktı eşitliğini doğrular.
Augment'te bilinçli olarak **hue oynaması yoktur** — hue, etiketin kendisidir.

Tek GPU'da dakikalar-onlarca dakika sürer; `--device mps` ile M-serisi Mac'te
bile eğitilebilir.

### 6.3 Bağlama

```yaml
attributes:
  color:
    method: model
    model_path: ./weights/color.onnx     # color.txt yan dosyası OTOMATIK yüklenir
```

---

## 7. Marka/model sınıflandırıcı

En dürüst uyarının bölümü: **bu alanda ticari-temiz hazır veri yok denecek
kadar azdır.**

| Kaynak | Durum |
|---|---|
| Stanford Cars | Resmi dağıtım linkleri ölü; Kaggle mirror'larının lisansı tanımsız. Satılabilir ürüne dayanak yapma. |
| CompCars | İmzalı akademik agreement ister; ticari kullanım kapsam dışı. |
| VMMRdb | Araştırma-sadece. |
| Kendi verin | Tek gerçek ticari-temiz yol: kendi snapshot'larını topla, marka_model klasörlerine ayır. Türkiye parkuru için 30–60 yaygın modelle başlamak hem gerçekçi hem işlevseldir. |

Bu yüzden `attributes.makemodel.enabled` platformda varsayılan `false`.

### 7.1 Veri düzeni

```
/data/datasets/makemodel/
  train/renault_clio/*.jpg
  train/ford_transit/*.jpg
  ...
  val/renault_clio/*.jpg
```

Klasör adı = dashboard'da görünecek etiket. `kucuk_harf_alt_cizgi` kullan.

### 7.2 Eğitim + ONNX export

```bash
python training/train_makemodel.py \
  --data /data/datasets/makemodel \
  --arch resnet50 --epochs 60 --batch 64 --label-smoothing 0.1 \
  --out training/runs/makemodel/makemodel.onnx
```

train_color ile aynı iskelet; farkları görevden gelir: iki-katmanlı daha derin
baş (Linear→GELU→Dropout→Linear), label smoothing 0.1 (yüzlerce benzer sınıfta
aşırı-özgüveni kırar), tam ColorJitter + RandomGrayscale (boya rengi
marka/model ipucu OLMAMALI) ve RandomErasing.

`--imgsz` değerini 224'te bırak: platformdaki `MakeModelExtractor` crop'ları
sabit 224×224 besler (renk modelinin aksine giriş boyutunu ONNX'ten okumaz);
farklı boyutla export edilen statik graf çalışma anında reddedilir.

### 7.3 Bağlama

```yaml
attributes:
  makemodel:
    enabled: true
    model_path: ./weights/makemodel.onnx
    labels_path: ./weights/makemodel.txt   # ZORUNLU — renk modelinin aksine
                                           # yan dosya otomatik YÜKLENMEZ
    min_confidence: 0.35
```

---

## 8. Sonuçları Panoptes'e bağlama

### 8.1 Araç dedektörü

Üç üretim yolu (hepsi `panoptes.yaml`'daki `detector` bloğu):

```yaml
# A) ONNX — önerilen: taşınabilir, sınıf isimleri metadata'dan otomatik gelir
detector:
  backend: onnx
  model: ./weights/vehicles.onnx
  imgsz: 640
  conf: 0.3
  device: auto
  # metadata'sız üçüncü-parti ONNX için isimleri elle ver:
  # extra:
  #   names: {0: car, 1: van, 2: bus, 3: truck, 4: motorcycle, 5: bicycle, 6: emergency, 7: person}

# B) TensorRT — en hızlı; engine'i deployment GPU'sunda üret (§3.4)
detector:
  backend: tensorrt
  model: ./weights/vehicles.engine
  imgsz: 640
  conf: 0.3

# C) Doğrudan .pt — yalnızca [yolo] extra kuruluysa (AGPL yükümlülükleri geçerli)
detector:
  backend: ultralytics
  model: ./weights/vehicles_best.pt
  imgsz: 640
  conf: 0.3
```

Onnx backend'i hem YOLO26'nın uçtan-uca `(B, 300, 6)` çıktısını hem klasik
`(B, 84, 8400)` düzenini tanır — export ettiğin model hangisiyse otomatik uyar.

**RF-DETR fine-tune'unu bağlama (D yolu):** rfdetr backend'i,
`detector.extra.rfdetr_kwargs` sözlüğünü model kurucusuna olduğu gibi
geçirir — fine-tune checkpoint'i `pretrain_weights` ile yüklenir:

```yaml
# D) RF-DETR fine-tune checkpoint'i (Apache-temiz üretim yolu)
detector:
  backend: rfdetr
  model: rfdetr-small            # eğitimde kullandığın tier ile aynı olmalı
  imgsz: 640
  conf: 0.3
  extra:
    rfdetr_kwargs:
      pretrain_weights: ./weights/rfdetr-ft.pth
```

Dürüst kalan tek boşluk sınıf isimleridir: rfdetr backend'i tespit id'lerini
bugün COCO-91 isim tablosu üzerinden yorumlar ve onnx backend'indeki
`extra.names` benzeri bir isim override'ı henüz yoktur. COCO dışı bir
sözlükle (örn. §2.4'teki kanonik 8-sınıf düzeni) fine-tune ettiysen id'ler
yanlış adlandırılır ya da düşer — backend'e isim override'ı eklenene kadar
D yolunu yalnızca COCO-sınıflı fine-tune'lar için kullan. Ayrıca RF-DETR'in
kendi ONNX çıktı düzeni YOLO-stili onnx backend'iyle uyumlu değildir;
A/B yolları ultralytics→ONNX export'una özgüdür.

### 8.2 ALPR (plaka dedektörü + OCR)

```yaml
alpr:
  enabled: true
  detector_model: yolo-v9-s-608-license-plate-end2end   # open-image-models hub adı
  ocr_model: cct-s-v2-global-model                      # fast-plate-ocr hub adı
  country: TR
```

**[SÜRÜM-KONTROL]** Bu iki alan doğrudan ilgili paketlerin kurucularına
geçirilir (`LicensePlateDetector(detection_model=...)`,
`LicensePlateRecognizer(...)`). Kendi eğittiğin modeli takmak istiyorsan
kurulu paket sürümünün hub adı yerine yerel dosya yolu/özel model kabul edip
etmediğini docs'tan doğrula; kabul etmiyorsa entegrasyon
`src/panoptes/alpr/detector.py` / `ocr.py`'de küçük bir uyarlama ister (bu
dosyalar tam da bu amaçla ince sarmalayıcı olarak yazıldı). Fine-tune OCR
modelini doğrulamadan üretime koyma — §5.3'teki karşılaştırmayı yap.

### 8.3 `deploy/weights_manifest.yaml` güncelleme

Ürüne girebilecek **her** ağırlık manifest'e girer — CI lisans kapısı bunu
denetler. Kendi eğittiğin model için şablon:

```yaml
  - id: vehicles-panoptes-v1
    family: yolo26            # veya rf-detr
    task: vehicle-detection
    file: vehicles.onnx
    source: internal-training/vehicles-y26s   # training/runs/ altındaki koşu adı
    spdx: AGPL-3.0-only       # YOLO26 fine-tune ise! RF-DETR fine-tune ise Apache-2.0
    ship_ok: false            # AGPL ise Enterprise License alınana dek false;
                              # RF-DETR + ticari-temiz veri ise true
    provenance: >-
      Fine-tuned from yolo26s.pt on COCO2017 (CC BY 4.0) + Roboflow
      vehicles-coco (CC BY 4.0) + in-house footage; 100 epochs @ 640.
    notes: Eğitim verisi katmanları EGITIM.md §2.1; atıf listesi NOTICE'ta.
```

`provenance` alanına eğitim verilerini eksiksiz yaz — CC BY atıf yükümlülüğünün
ve KVKK sorulabilirliğinin kaydı burasıdır.

### 8.4 Hız doğrulaması (benchmark)

```bash
panoptes benchmark --backend onnx --model ./weights/vehicles.onnx --imgsz 640 --frames 400 --batch 8
panoptes benchmark --backend tensorrt --model ./weights/vehicles.engine --imgsz 640 --frames 400 --batch 8
```

Çıktı FPS + p50/p95 gecikme verir. Akış başına hedef bütçen:
`aktif_fps × akış_sayısı < ölçülen FPS` (governor boş sahnede zaten
`idle_fps`'e düşer).

Uçtan uca duman testi:

```bash
panoptes process /data/raw/ornek-video.mp4 --config panoptes.yaml
panoptes demo   # model indirmeden, mock dedektörle boru hattının sağlığı
```

---

## 9. Sorun giderme

### 9.1 CUDA out of memory

- Ultralytics: `--batch -1` zaten VRAM'e göre otomatik seçer; elle verdiysen yarıla.
- Yine OOM: `--imgsz 640 → 512`, en son model boyunu küçült (`s → n`).
- RF-DETR: `--batch 2 --grad-accum 8` (etkin batch aynı kalır, tepe VRAM düşer).
- Sınıflandırıcılar: `--batch 32`; resnet50 → resnet18/efficientnet_v2_s.
- Eğitim sırasında `nvidia-smi`'da başka süreç var mı bak (kiralık makinelerde
  önceki kiracının zombisi kalabiliyor).

### 9.2 GPU aç kalıyor (dataloader darboğazı)

Belirti: `nvtop`'ta GPU %40'ın altında dalgalanıyor, epoch süreleri disk hızına
takılı.

- Script'ler `cache="disk"` geçirir — ilk epoch yavaş, sonrası hızlanır; bu normal.
- `--workers` değerini vCPU sayına yaklaştır (varsayılan 16).
- Veri /data'da (NVMe) mi? Ağ diskinden eğitme.
- RAM boysa `cache="disk"` yerine `cache=True` (RAM) daha da hızlıdır —
  `train_detector.py`'de tek satır değişiklik; 100k+ görüntüde RAM'e sığmaz,
  elleme.

### 9.3 Çoklu GPU

Ultralytics DDP'yi kendisi yönetir: `--device 0,1`. Etkin batch GPU sayısıyla
çarpılır; LR'ı elle ölçekleme, `batch=-1` + varsayılan LR politikası bunu
karşılar. RF-DETR çoklu-GPU davranışı **[SÜRÜM-KONTROL]** — 1.8.x docs'una bak;
tek büyük GPU (L40S) genelde daha az dertlidir.

### 9.4 KVKK hatırlatması (kendi verinle eğitim)

Plaka görüntüsü ve plaka metni **kişisel veridir** (KVKK; AB tarafında GDPR).
Kendi kameralarından topladığın veriyle eğitim yapmadan önce:

- **Hukuki dayanağını belgele** — kendi tesisinde güvenlik amaçlı çekim çoğu
  durumda meşru menfaate dayanır ama *eğitim amaçlı yeniden kullanım* ayrı bir
  işleme amacıdır; aydınlatma metnine ekle, VERBİS kaydını güncelle,
  gerekiyorsa etki değerlendirmesi (DPIA) yap.
- **Veri minimizasyonu:** eğitim setine tam kare değil crop al (plaka
  dedektörü/OCR için `plate_crop.py` zaten bunu yapar); insan yüzü içeren
  kareleri ayıkla veya bulanıklaştır.
- **Saklama süresi:** ham kayıtlara TTL koy; Panoptes'in
  `privacy.snapshot_retention_days` ve `database.retention_days` alanları
  üretim tarafını çözer ama eğitim kopyaları senin sorumluluğunda.
- Eğitilmiş modelin kendisi tipik olarak kişisel veri içermez, fakat eğitim
  setinin hangi hukuki dayanakla toplandığı sorulabilir — `weights_manifest.yaml`
  `provenance` alanına yaz (§8.3).
- Bu bölüm hukuki tavsiye değildir; müşteri sözleşmesi öncesi KVKK danışmanına
  doğrulat.

### 9.5 Sık düşülen tuzaklar

| Belirti | Sebep / çözüm |
|---|---|
| `ModuleNotFoundError: ultralytics` | venv aktif değil (`source .venv/bin/activate`) veya `[train]` extra kurulmadı |
| Roboflow indirme `401` | `ROBOFLOW_API_KEY` bu shell'de export edilmemiş |
| YOLO eğitimi başlarken `train: labels not found` | images/labels dizin eşleşmesi bozuk — ultralytics `images/` yolundan `labels/` türetir; §2.3'teki düzeni birebir kur |
| Sınıf-bazlı mAP'te van/emergency = 0 | O sınıflara hiç örnek girmedi (§2.4) — hata değil, veri boşluğu |
| ONNX modeli platformda sınıfları yanlış adlandırıyor | Üçüncü-parti export'ta `names` metadata'sı yok — `detector.extra.names` ver (§8.1) |
| TensorRT engine başka makinede açılmıyor | Engine GPU mimarisine özgüdür — deployment GPU'sunda yeniden export et |
| RTX 5090'da `no kernel image is available` | PyPI torch wheel'i sm_120 içermiyor — cu128 index'iyle kur (§1.3) |
| Fine-tune OCR global modelden kötü | Eğitim etiketleri kirli (ön-doldurmayı elden geçirmedin) veya sentetik/gerçek oranı aşırı sentetik — §5.1/§5.3 |
