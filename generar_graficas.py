"""
Genera las figuras del informe final a partir de los modelos ya entrenados.
No modifica taller_calidad_frutas_bueno_regular_malo1.ipynb ni app/app.py.

Las figuras se guardan en outputs/figures/ como PDF vectoriales.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
import joblib

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})

OUTPUT_DIR = Path("outputs")
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generando figuras para el informe final")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Carga de datos (misma lógica que el notebook)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
CLASS_NAMES = ["bueno", "regular", "malo"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\n[1/9] Cargando dataset...")
rows = []
for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name
    if class_dir.exists():
        for file_path in class_dir.rglob("*"):
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"filepath": str(file_path), "label": class_name})

images_df = pd.DataFrame(rows)
print(f"  Imágenes encontradas: {len(images_df)}")

# ---------------------------------------------------------------------------
# 2. Train/val/test split (misma semilla, mismo orden)
# ---------------------------------------------------------------------------
print("[2/9] Dividiendo en train/val/test...")
X_paths = images_df["filepath"].to_numpy()
y_labels = images_df["label"].to_numpy()

X_train_paths, X_temp_paths, y_train, y_temp = train_test_split(
    X_paths, y_labels, test_size=0.30, random_state=42, stratify=y_labels,
)
X_val_paths, X_test_paths, y_val, y_test = train_test_split(
    X_temp_paths, y_temp, test_size=0.50, random_state=42, stratify=y_temp,
)

# Balanceo por subcarpeta (idéntico al notebook)
train_df = pd.DataFrame({"filepath": X_train_paths, "label": y_train})
parent_dirs = set(Path(p).parent.name for p in train_df["filepath"])
if parent_dirs:
    for cls in CLASS_NAMES:
        cls_mask = train_df["label"] == cls
        sub_names = train_df.loc[cls_mask, "filepath"].apply(lambda p: Path(p).parent.name)
        sub_counts = sub_names.value_counts()
        if len(sub_counts) <= 1:
            continue
        median_count = int(sub_counts.median())
        max_allowed = max(int(median_count * 1.5), 1)
        keep = []
        for sub_name in sub_counts.index:
            sub_mask = cls_mask & (train_df["filepath"].apply(lambda p: Path(p).parent.name) == sub_name)
            subset = train_df[sub_mask]
            if len(subset) > max_allowed:
                subset = subset.sample(n=max_allowed, random_state=42)
            keep.append(subset)
        train_df = pd.concat([train_df[~cls_mask]] + keep)
X_train_paths = train_df["filepath"].to_numpy()
y_train = train_df["label"].to_numpy()

# ---------------------------------------------------------------------------
# 3. Extracción de características (misma función que el notebook)
# ---------------------------------------------------------------------------
print("[3/9] Extrayendo características...")

IMG_SIZE = (64, 64)
HIST_BINS = 32


def resize_with_padding(image, target_size, bg_color=(255, 255, 255)):
    orig_w, orig_h = image.size
    aspect = orig_w / orig_h
    tw, th = target_size
    if aspect > 1:
        nw, nh = tw, int(tw / aspect)
    else:
        nh, nw = th, int(th * aspect)
    resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def load_image_array(path, size=IMG_SIZE):
    image = Image.open(path).convert("RGB")
    image = resize_with_padding(image, size)
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_features(path):
    arr = load_image_array(path)
    hist_features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel].ravel(), bins=HIST_BINS, range=(0.0, 1.0))
        hist_features.append(hist)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    return np.concatenate([np.concatenate(hist_features), mean, std])


X_test_feat = np.vstack([extract_features(p) for p in X_test_paths])
label_encoder = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
y_test_enc = label_encoder.transform(y_test)
classes = label_encoder.classes_
n_classes = len(classes)
print(f"  Test set: {len(X_test_paths)} imágenes")

# ---------------------------------------------------------------------------
# 4. Cargar modelo Random Forest
# ---------------------------------------------------------------------------
print("[4/9] Cargando modelo Random Forest...")
rf_path = OUTPUT_DIR / "best_traditional_model_random_forest.pkl"
if not rf_path.exists():
    raise FileNotFoundError(f"No se encuentra {rf_path}. Ejecuta el notebook primero.")
rf_model = joblib.load(rf_path)

test_pred = rf_model.predict(X_test_feat)

prec_per = precision_score(y_test_enc, test_pred, average=None, zero_division=0)
rec_per = recall_score(y_test_enc, test_pred, average=None, zero_division=0)
f1_per = f1_score(y_test_enc, test_pred, average=None)
acc = accuracy_score(y_test_enc, test_pred)
f1_macro = f1_score(y_test_enc, test_pred, average="macro")

# ---------------------------------------------------------------------------
# 5. FIGURA 1 — Distribución de clases
# ---------------------------------------------------------------------------
print("[5/9] Figura 1: Distribución de clases...")
fig, ax = plt.subplots(figsize=(5, 3))
counts = images_df["label"].value_counts().reindex(CLASS_NAMES, fill_value=0)
bars = ax.bar(CLASS_NAMES, counts.values,
              color=["#4CAF50", "#FFB300", "#E53935"], edgecolor="white", linewidth=0.5)
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{count:,}\n({100*count/len(images_df):.1f}%)",
            ha="center", va="bottom", fontsize=8)
ax.set_title("Distribución de clases en el dataset", fontsize=11, fontweight="bold")
ax.set_ylabel("Cantidad de imágenes")
ax.set_xlabel("Clase de calidad")
ax.margins(y=0.15)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_distribucion.pdf")
plt.close(fig)
print("  -> fig_distribucion.pdf")

# ---------------------------------------------------------------------------
# 6. FIGURA 2 — Matriz de confusión RF
# ---------------------------------------------------------------------------
print("[6/9] Figura 2: Matriz de confusión RF...")
cm = confusion_matrix(y_test_enc, test_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Matriz de confusión — Random Forest", fontsize=11, fontweight="bold")
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_matriz_confusion_rf.pdf")
plt.close(fig)
print("  -> fig_matriz_confusion_rf.pdf")

# ---------------------------------------------------------------------------
# 7. FIGURA 3 — Métricas por clase RF
# ---------------------------------------------------------------------------
print("[7/9] Figura 3: Métricas por clase RF...")
x = np.arange(n_classes)
width = 0.25
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(x - width, prec_per, width, label="Precisión", color="#4CAF50", edgecolor="white")
ax.bar(x, rec_per, width, label="Recall", color="#2196F3", edgecolor="white")
ax.bar(x + width, f1_per, width, label="F1-score", color="#FF9800", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("Puntuación")
ax.set_title("Métricas por clase — Random Forest", fontsize=11, fontweight="bold")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
for i in range(n_classes):
    ax.text(i - width, prec_per[i] + 0.02, f"{prec_per[i]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i, rec_per[i] + 0.02, f"{rec_per[i]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i + width, f1_per[i] + 0.02, f"{f1_per[i]:.3f}", ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_metricas_clase_rf.pdf")
plt.close(fig)
print("  -> fig_metricas_clase_rf.pdf")

# ---------------------------------------------------------------------------
# 8. FIGURA 4 — Curvas ROC RF
# ---------------------------------------------------------------------------
print("[8/9] Figura 4: Curvas ROC RF...")
if hasattr(rf_model, "predict_proba"):
    y_prob = rf_model.predict_proba(X_test_feat)
    y_test_bin = pd.get_dummies(y_test_enc).values
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors_roc = ["#4CAF50", "#FF9800", "#E53935"]
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors_roc[i],
                label=f"{classes[i]} (AUC = {roc_auc:.3f})", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    ax.set_title("Curvas ROC — Random Forest", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_roc_rf.pdf")
    plt.close(fig)
    print("  -> fig_roc_rf.pdf")

# ---------------------------------------------------------------------------
# 9. FIGURA 5 — Importancia de características RF
# ---------------------------------------------------------------------------
print("[9/9] Figura 5: Importancia de características RF...")
rf_inner = rf_model.named_steps["classifier"] if hasattr(rf_model, "named_steps") else rf_model
if hasattr(rf_inner, "feature_importances_"):
    importances = rf_inner.feature_importances_
    top_n = min(15, len(importances))
    top_idx = np.argsort(importances)[-top_n:]
    labels = []
    for idx in top_idx:
        if idx < 96:
            ch = idx // 32
            bin_ = idx % 32
            labels.append(f"Hist C{ch+1} B{bin_+1}")
        elif idx < 99:
            labels.append(f"Media C{idx-95}")
        else:
            labels.append(f"Std C{idx-98}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(top_n), importances[top_idx], color="#2196F3", edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Importancia relativa")
    ax.set_title("Top 15 características — Random Forest", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_importancia_rf.pdf")
    plt.close(fig)
    print("  -> fig_importancia_rf.pdf")

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Figuras generadas en:", FIG_DIR.resolve())
print("=" * 60)
for f in sorted(FIG_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name} ({size_kb:.1f} KB)")
