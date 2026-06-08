# -*- coding: utf-8 -*-
"""
Drone vs Bird Görüntü Sınıflandırması
======================================
BLM0463 - Veri Madenciliğine Giriş | Dönem Projesi
Yöntem        : CNN (MobileNetV2 Transfer Learning) + HOG+SVM Karşılaştırması
Veri Seti     : Açık erişimli drone-kuş görüntü veri seti
Çalışma Ortamı: Python 3.10+ | GPU önerilir (Google Colab veya yerel)

Kullanım:
  Veri setini dataset/BirdVsDrone/ klasörüne yerleştirip scripti çalıştırın.
  Beklenen yapı:
    dataset/BirdVsDrone/Birds/   <- kuş görüntüleri
    dataset/BirdVsDrone/Drones/  <- drone görüntüleri
"""

# ─────────────────────────────────────────────────────────
# 0. VERİ SETİ KLASÖR KONTROLÜ
# ─────────────────────────────────────────────────────────
import os
import glob

DATASET_ROOT = os.path.join('dataset', 'BirdVsDrone')

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(
        f'Veri seti klasörü bulunamadı: {DATASET_ROOT}\n'
        'Lütfen veri setini dataset/BirdVsDrone/ altına yerleştirin.\n'
        'Beklenen yapı:\n'
        '  dataset/BirdVsDrone/Birds/   <- kuş görüntüleri\n'
        '  dataset/BirdVsDrone/Drones/  <- drone görüntüleri'
    )

# Klasör yapısını kontrol et
print('Veri seti yapısı:')
for root_dir, dirs, files_list in os.walk(DATASET_ROOT):
    level = root_dir.replace(DATASET_ROOT, '').count(os.sep)
    if level < 3:
        n = len(glob.glob(os.path.join(root_dir, '*.*')))
        print('  ' * level + os.path.basename(root_dir) + f'/  ({n} dosya)')

# ─────────────────────────────────────────────────────────
# 1. KÜTÜPHANELER & SABITLER
# ─────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, glob, cv2, warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc, f1_score)
from skimage.feature import hog

# Hiperparametreler
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 15
SEED       = 42
TRAIN_DIR  = DATASET_ROOT
classes    = ['Drones', 'Birds']

tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f'TF  : {tf.__version__}')
print(f'GPU : {tf.config.list_physical_devices("GPU")}')

# ─────────────────────────────────────────────────────────
# 2. KEŞİFSEL VERİ ANALİZİ (EDA)
# ─────────────────────────────────────────────────────────
counts = {cls: len(glob.glob(os.path.join(TRAIN_DIR, cls, '*.*'))) for cls in classes}
print('Sınıf dağılımı:', counts)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = ['#2196F3', '#4CAF50']

bars = axes[0].bar(counts.keys(), counts.values(), color=colors, edgecolor='white')
axes[0].set_title('Sınıf Dağılımı', fontweight='bold')
axes[0].set_ylabel('Görüntü Sayısı')
for bar, val in zip(bars, counts.values()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', fontweight='bold')

axes[1].pie(counts.values(), labels=counts.keys(), colors=colors,
            autopct='%1.1f%%', startangle=90)
axes[1].set_title('Sınıf Oranı', fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Örnek görüntüler
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
fig.suptitle('Örnek Görüntüler', fontsize=14, fontweight='bold')

for row, cls in enumerate(classes):
    img_paths = glob.glob(os.path.join(TRAIN_DIR, cls, '*.*'))[:5]
    for col, path in enumerate(img_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        axes[row, col].imshow(img)
        axes[row, col].set_title(cls, fontsize=9)
        axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# 3. VERİ YÜKLEME & ARTIRMA (ImageDataGenerator)
# ─────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    seed=SEED,
    shuffle=False
)

CLASS_INDICES = train_gen.class_indices
print('Sınıf indeksleri:', CLASS_INDICES)
print(f'Train: {train_gen.samples} | Val: {val_gen.samples}')

# ─────────────────────────────────────────────────────────
# 4. MODEL: MobileNetV2 TRANSFER LEARNING
# ─────────────────────────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

print(f'Toplam katman    : {len(model.layers)}')
print(f'Eğitilebilir     : {sum(1 for l in model.layers if l.trainable)}')

# ─── Aşama 1: Üst katmanları eğit ───
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# ─── Aşama 2: Fine-tuning ───
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

callbacks2 = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks2,
    verbose=1
)

# ─────────────────────────────────────────────────────────
# 5. EĞİTİM GRAFİKLERİ
# ─────────────────────────────────────────────────────────
h = {}
for key in history1.history:
    h[key] = history1.history[key] + history2.history[key]

ep    = range(1, len(h['accuracy']) + 1)
split = len(history1.history['accuracy'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MobileNetV2 Eğitim Süreci', fontweight='bold')

axes[0].plot(ep, h['accuracy'], 'b-o', ms=4, label='Train')
axes[0].plot(ep, h['val_accuracy'], 'r-o', ms=4, label='Validation')
axes[0].axvline(split, color='gray', ls='--', alpha=0.7, label='Fine-tune başlangıcı')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, h['loss'], 'b-o', ms=4, label='Train')
axes[1].plot(ep, h['val_loss'], 'r-o', ms=4, label='Validation')
axes[1].axvline(split, color='gray', ls='--', alpha=0.7, label='Fine-tune başlangıcı')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# 6. CNN DEĞERLENDİRME METRİKLERİ
# ─────────────────────────────────────────────────────────
val_gen.reset()
y_pred_prob = model.predict(val_gen, verbose=1).flatten()
y_pred      = (y_pred_prob > 0.5).astype(int)
y_true      = val_gen.classes
class_names = list(CLASS_INDICES.keys())

cm           = confusion_matrix(y_true, y_pred)
TN, FP, FN, TP = cm.ravel()

acc     = accuracy_score(y_true, y_pred)
sens    = TP / (TP + FN)
spec    = TN / (TN + FP)
prec    = TP / (TP + FP)
f1      = f1_score(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

print('─' * 40)
print(f'Accuracy    : {acc:.4f}')
print(f'Sensitivity : {sens:.4f}')
print(f'Specificity : {spec:.4f}')
print(f'Precision   : {prec:.4f}')
print(f'F1-Score    : {f1:.4f}')
print(f'AUC         : {roc_auc:.4f}')
print('─' * 40)
print(classification_report(y_true, y_pred, target_names=class_names))

cnn_metrics = {
    'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
    'Precision': prec, 'F1-Score': f1, 'AUC': roc_auc
}

# CNN görsel sonuçlar
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('CNN — Confusion Matrix')
axes[0].set_xlabel('Tahmin'); axes[0].set_ylabel('Gerçek')

cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('CNN — CM (%)')
axes[1].set_xlabel('Tahmin')

axes[2].plot(fpr, tpr, 'b-', lw=2.5, label=f'CNN AUC={roc_auc:.3f}')
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[2].fill_between(fpr, tpr, alpha=0.1, color='blue')
axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR')
axes[2].set_title('ROC Eğrisi'); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# 7. HOG + SVM
# ─────────────────────────────────────────────────────────
print('HOG özellikleri çıkarılıyor...')
HOG_SIZE = 128
X_hog, y_hog = [], []

for cls_name, cls_idx in CLASS_INDICES.items():
    cls_path  = os.path.join(TRAIN_DIR, cls_name)
    img_paths = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]
    print(f'  {cls_name}: {len(img_paths)} dosya')
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img  = cv2.resize(img, (HOG_SIZE, HOG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)
        X_hog.append(feat)
        y_hog.append(cls_idx)

X_hog = np.array(X_hog)
y_hog = np.array(y_hog)
print(f'HOG özellik vektörü boyutu: {X_hog.shape}')

X_tr, X_te, y_tr, y_te = train_test_split(
    X_hog, y_hog, test_size=0.2, random_state=SEED, stratify=y_hog
)
scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr)
X_te   = scaler.transform(X_te)

print('SVM eğitiliyor (RBF kernel, C=10)...')
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=SEED)
svm.fit(X_tr, y_tr)
print('✅ SVM eğitimi tamamlandı')

# SVM değerlendirme
svm_pred = svm.predict(X_te)
svm_prob = svm.predict_proba(X_te)[:, 1]

svm_cm           = confusion_matrix(y_te, svm_pred)
TN2, FP2, FN2, TP2 = svm_cm.ravel()

s_acc  = accuracy_score(y_te, svm_pred)
s_sens = TP2 / (TP2 + FN2)
s_spec = TN2 / (TN2 + FP2)
s_prec = TP2 / (TP2 + FP2)
s_f1   = f1_score(y_te, svm_pred)
fpr2, tpr2, _ = roc_curve(y_te, svm_prob)
s_auc  = auc(fpr2, tpr2)

print('─' * 40)
print(f'SVM Accuracy    : {s_acc:.4f}')
print(f'SVM Sensitivity : {s_sens:.4f}')
print(f'SVM Specificity : {s_spec:.4f}')
print(f'SVM Precision   : {s_prec:.4f}')
print(f'SVM F1-Score    : {s_f1:.4f}')
print(f'SVM AUC         : {s_auc:.4f}')
print('─' * 40)

svm_metrics = {
    'Accuracy': s_acc, 'Sensitivity': s_sens, 'Specificity': s_spec,
    'Precision': s_prec, 'F1-Score': s_f1, 'AUC': s_auc
}

# ─────────────────────────────────────────────────────────
# 8. CNN vs HOG+SVM KARŞILAŞTIRMA GRAFİKLERİ
# ─────────────────────────────────────────────────────────
metrics_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC']
cnn_vals = [cnn_metrics[m] for m in metrics_names]
svm_vals = [svm_metrics[m] for m in metrics_names]

x = np.arange(len(metrics_names))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('CNN (MobileNetV2) vs HOG+SVM Karşılaştırması', fontsize=14, fontweight='bold')

b1 = axes[0].bar(x - w / 2, cnn_vals, w, label='CNN (MobileNetV2)', color='#2196F3', alpha=0.85)
b2 = axes[0].bar(x + w / 2, svm_vals, w, label='HOG+SVM', color='#FF5722', alpha=0.85)
for bar in list(b1) + list(b2):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}', ha='center', fontsize=8, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_names, rotation=15)
axes[0].set_ylim([0, 1.12])
axes[0].legend()
axes[0].grid(alpha=0.3, axis='y')
axes[0].set_title('Metrik Karşılaştırması')

axes[1].plot(fpr, tpr, 'b-', lw=2.5, label=f'CNN (AUC={roc_auc:.3f})')
axes[1].plot(fpr2, tpr2, color='#FF5722', lw=2.5, label=f'SVM (AUC={s_auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].fill_between(fpr, tpr, alpha=0.08, color='blue')
axes[1].fill_between(fpr2, tpr2, alpha=0.08, color='#FF5722')
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Karşılaştırması')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ comparison.png kaydedildi')

# ─────────────────────────────────────────────────────────
# 9. ÇIKTILARI KAYDET
# ─────────────────────────────────────────────────────────
os.makedirs('outputs', exist_ok=True)

model.save(os.path.join('outputs', 'drone_bird_model.keras'))
print('✅ Model kaydedildi: outputs/drone_bird_model.keras')

pngs = ['class_distribution.png', 'sample_images.png',
        'training_curves.png', 'cnn_results.png', 'comparison.png']

print('\nÜretilen çıktı görselleri:')
for f in pngs:
    if os.path.exists(f):
        dest = os.path.join('outputs', f)
        import shutil
        shutil.move(f, dest)
        print(f'  ✅ outputs/{f}')
    else:
        print(f'  ❌ bulunamadı: {f}')

print('\n✅ Tüm çıktılar outputs/ klasörüne kaydedildi.')
