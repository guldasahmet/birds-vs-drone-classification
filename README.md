# Drone vs Bird Görüntü Sınıflandırması

BLM0463 — Veri Madenciliğine Giriş dersi kapsamında gerçekleştirilen dönem projesidir.  
Açık erişimli bir drone-kuş görüntü veri seti üzerinde iki farklı sınıflandırma yöntemi uygulanmış ve sonuçlar karşılaştırılmıştır.

---

## İçindekiler

- [Projenin Amacı](#projenin-amacı)
- [Kullanılan Yöntemler](#kullanılan-yöntemler)
- [Veri Seti](#veri-seti)
- [Kurulum ve Çalıştırma](#kurulum-ve-çalıştırma)
- [Sonuç Metrikleri](#sonuç-metrikleri)
- [Proje Çıktıları](#proje-çıktıları)
- [Gelecek Çalışma Önerileri](#gelecek-çalışma-önerileri)

---

## Projenin Amacı

Bu proje, drone ve kuş görüntülerini otomatik olarak birbirinden ayırt edebilen bir ikili sınıflandırıcı geliştirmeyi amaçlamaktadır. Çalışmada derin öğrenme tabanlı bir yöntem (MobileNetV2 transfer learning) ile klasik makine öğrenmesi tabanlı bir yöntem (HOG özellik çıkarımı + SVM) karşılaştırmalı olarak ele alınmıştır.

---

## Kullanılan Yöntemler

### 1. MobileNetV2 Transfer Learning

ImageNet üzerinde önceden eğitilmiş MobileNetV2 ağı temel model olarak kullanılmıştır. Eğitim iki aşamada gerçekleştirilmiştir:

- **Aşama 1 — Freeze**: Temel model dondurulur, yalnızca üst katmanlar eğitilir (`lr = 1e-3`, 10 epoch).
- **Aşama 2 — Fine-tuning**: Son 30 katman açılarak düşük öğrenme hızıyla ince ayar yapılır (`lr = 1e-5`, 15 epoch).

| Parametre | Değer |
|---|---|
| Temel Model | MobileNetV2 (ImageNet ağırlıkları) |
| Giriş Boyutu | 224 × 224 × 3 |
| Optimizör | Adam |
| Kayıp Fonksiyonu | Binary Crossentropy |
| Dropout | 0.4 + 0.3 |
| Veri Artırma | Döndürme ±20°, kaydırma ±20%, zoom ±20%, yatay çevirme |

### 2. HOG + SVM

Görüntülerden Histogram of Oriented Gradients (HOG) yöntemiyle özellik vektörleri çıkarılmış, ardından RBF çekirdekli Destek Vektör Makinesi ile sınıflandırma yapılmıştır.

| Parametre | Değer |
|---|---|
| HOG Görüntü Boyutu | 128 × 128 |
| Yönelim Sayısı | 9 |
| Piksel / Hücre | 8 × 8 |
| Hücre / Blok | 2 × 2 |
| SVM Çekirdeği | RBF |
| C | 10 |
| Gamma | scale |

---

## Veri Seti

Çalışmada açık erişimli bir drone-kuş görüntü veri seti kullanılmıştır. Veri seti iki sınıftan oluşmaktadır:

- `Birds` — Kuş görüntüleri
- `Drones` — Drone görüntüleri

Tüm görüntüler RGB formatında olup model girişi için 224×224 piksel boyutuna yeniden boyutlandırılmıştır. Veri %80 eğitim / %20 doğrulama olarak bölünmüştür.

Veri setini edindikten sonra aşağıdaki klasör yapısında yerleştirin:

```
dataset/
└── BirdVsDrone/
    ├── Birds/
    │   ├── img001.jpg
    │   └── ...
    └── Drones/
        ├── img001.jpg
        └── ...
```

---

## Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

### Çalıştırma

1. Veri setini `dataset/BirdVsDrone/` klasörüne yerleştirin (yukarıdaki yapıya uygun şekilde).
2. Aşağıdaki komutu çalıştırın:

```bash
python notebooks/birds_vs_drone_full.py
```

> **Not:** GPU ortamı önerilir. Google Colab üzerinde de çalıştırılabilir — bu durumda `TRAIN_DIR` değişkenini Colab yoluna (`/content/dataset/BirdVsDrone`) göre güncelleyin.

### Kütüphaneler

```
tensorflow >= 2.12
scikit-learn >= 1.3
scikit-image >= 0.21
opencv-python >= 4.8
matplotlib >= 3.7
seaborn >= 0.12
numpy >= 1.24
```

---

## Sonuç Metrikleri

| Metrik | MobileNetV2 | HOG + SVM |
|---|:-:|:-:|
| Accuracy | **%93.29** | %84.94 |
| Sensitivity | %87.06 | — |
| Specificity | %100.00 | — |
| Precision | %100.00 | — |
| F1-Score | **%93.08** | %85.38 |
| AUC | **0.9911** | 0.9371 |

MobileNetV2, tüm metriklerde HOG+SVM yöntemini belirgin biçimde geride bırakmıştır. Özellikle AUC değeri (0.9911), modelin iki sınıfı ayırt etme kapasitesinin oldukça yüksek olduğunu göstermektedir.

---

## Proje Çıktıları

Aşağıdaki görseller `outputs/` klasöründe yer almaktadır:

| Dosya | Açıklama |
|---|---|
| `class_distribution.png` | Sınıf dağılımı (bar + pasta grafik) |
| `sample_images.png` | Her sınıftan örnek görüntüler |
| `training_curves.png` | Eğitim/doğrulama accuracy ve loss eğrileri |
| `cnn_results.png` | CNN confusion matrix ve ROC eğrisi |
| `comparison.png` | CNN vs HOG+SVM metrik ve ROC karşılaştırması |

---

## Gelecek Çalışma Önerileri

- **Farklı ağ mimarileri** denenebilir (EfficientNetB0, ResNet50, DenseNet).
- **Attention mekanizmaları** eklenerek model yorumlanabilirliği artırılabilir.
- **Veri artırma** stratejileri genişletilebilir (renk jitter, CutMix, Mixup).
- **Gerçek zamanlı çıkarım** için model optimize edilip ONNX/TFLite formatına dönüştürülebilir.
- Daha büyük ve çeşitli bir veri seti kullanılarak **genelleştirme kapasitesi** test edilebilir.

---

*BLM0463 Veri Madenciliğine Giriş — Dönem Projesi*
