import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import joblib
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_MEJORADO_DIR = Path(__file__).resolve().parent.parent / "outputs_mejorado"

st.set_page_config(
    page_title="Clasificación de Calidad de Frutas",
    page_icon="🍎",
    layout="centered",
)

# ===========================================================================
# SELECCION DE MODELO
# ===========================================================================
# Descomenta SOLO UNA de las siguientes lineas para elegir el modelo:
#
#   "base"          - Random Forest (tradicional, 32x32)
#   "mejorado"      - XGBoost (tradicional, 64x64)
#   "base_cnn"      - CNN (32x32)
#   "mejorado_cnn"  - CNN (64x64)

#MODEL_TYPE = "base_cnn"
#MODEL_TYPE = "base"
#MODEL_TYPE = "mejorado"
MODEL_TYPE = "mejorado_cnn"

# ===========================================================================
# MODEL LOADING
# ===========================================================================
@st.cache_resource
def cargar_modelo():
    if MODEL_TYPE == "base":
        return {
            "modelo": joblib.load(OUTPUT_DIR / "best_traditional_model.pkl"),
            "encoder": joblib.load(OUTPUT_DIR / "label_encoder.pkl"),
            "config": joblib.load(OUTPUT_DIR / "feature_config.pkl"),
            "tipo": "base",
            "arquitectura": "tradicional",
        }
    if MODEL_TYPE == "mejorado":
        cfg = joblib.load(OUTPUT_MEJORADO_DIR / "feature_config.pkl")
        return {
            "modelo": joblib.load(OUTPUT_MEJORADO_DIR / "best_traditional_model_mejorado.pkl"),
            "encoder": joblib.load(OUTPUT_MEJORADO_DIR / "label_encoder.pkl"),
            "config": cfg,
            "tipo": "mejorado",
            "arquitectura": "tradicional",
            "size_percentiles": cfg.get("size_percentiles"),
        }
    if MODEL_TYPE == "base_cnn":
        from tensorflow import keras
        return {
            "modelo": keras.models.load_model(OUTPUT_DIR / "cnn_model.keras"),
            "encoder": joblib.load(OUTPUT_DIR / "label_encoder.pkl"),
            "config": joblib.load(OUTPUT_DIR / "feature_config.pkl"),
            "tipo": "base_cnn",
            "arquitectura": "cnn",
        }
    if MODEL_TYPE == "mejorado_cnn":
        from tensorflow import keras
        cfg = joblib.load(OUTPUT_MEJORADO_DIR / "feature_config.pkl")
        return {
            "modelo": keras.models.load_model(OUTPUT_MEJORADO_DIR / "cnn_model_mejorado.keras"),
            "encoder": joblib.load(OUTPUT_MEJORADO_DIR / "label_encoder.pkl"),
            "config": cfg,
            "tipo": "mejorado_cnn",
            "arquitectura": "cnn",
            "size_percentiles": cfg.get("size_percentiles"),
        }
    raise ValueError("MODEL_TYPE invalido: " + MODEL_TYPE)


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
def resize_with_padding(img, target_size, bg_color=(255, 255, 255)):
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h
    tw, th = target_size
    if aspect > 1:
        nw, nh = tw, int(tw / aspect)
    else:
        nh, nw = th, int(th * aspect)
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target_size, bg_color)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas

def preprocesar_tradicional_base(imagen: Image.Image):
    TARGET_SIZE = (32, 32)
    HIST_BINS = 32
    img = resize_with_padding(imagen, TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    features = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c].ravel(), bins=HIST_BINS, range=(0.0, 1.0))
        features.append(hist)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    return np.concatenate([np.concatenate(features), mean, std]).reshape(1, -1)

def preprocesar_tradicional_mejorado(imagen: Image.Image):
    IMG_SIZE = (64, 64)
    HIST_BINS = 32
    img = resize_with_padding(imagen, IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hist_features = []
    for channel in range(3):
        channel_values = arr[:, :, channel].ravel()
        hist, _ = np.histogram(channel_values, bins=HIST_BINS, range=(0.0, 1.0))
        hist_features.append(hist)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    img_uint8 = (arr * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    texture_hist, _ = np.histogram(mag.ravel(), bins=16, range=(0, 255))
    texture_hist = texture_hist.astype(np.float32) / (texture_hist.sum() + 1e-8)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges) / 255.0
    img_array = np.asarray(imagen.convert("RGB"))
    gray_full = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray_full, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_px = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(largest))
    normalized_area = area_px / (IMG_SIZE[0] * IMG_SIZE[1])
    return np.concatenate([
        np.concatenate(hist_features), mean, std, texture_hist,
        [edge_density, normalized_area],
    ]).reshape(1, -1)

def preprocesar_cnn(imagen: Image.Image, size):
    arr = np.asarray(imagen.resize(size), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocesar_imagen(imagen: Image.Image, modelo_dict):
    t = modelo_dict["tipo"]
    if t == "base":
        return preprocesar_tradicional_base(imagen)
    if t == "mejorado":
        return preprocesar_tradicional_mejorado(imagen)
    if t == "base_cnn":
        return preprocesar_cnn(imagen, (32, 32))
    if t == "mejorado_cnn":
        return preprocesar_cnn(imagen, (96, 96))
    raise ValueError("tipo invalido: " + t)


# ---------------------------------------------------------------------------
# SIZE ESTIMATION
# ---------------------------------------------------------------------------
def estimar_tamano_base(imagen: Image.Image):
    arr = np.asarray(imagen.convert("RGB"), dtype=np.float32)
    threshold = 240
    mask = np.all(arr > threshold, axis=2)
    total = arr.shape[0] * arr.shape[1]
    fruit_pixels = total - np.sum(mask)
    ratio = fruit_pixels / total
    if ratio < 0.15:
        return "Pequeño", ratio
    elif ratio < 0.35:
        return "Mediano", ratio
    else:
        return "Grande", ratio

def estimar_tamano_mejorado(imagen: Image.Image, percentiles):
    img_array = np.asarray(imagen.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "Desconocido", 0.0
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    p33, p66 = percentiles
    if area <= p33:
        label = "Pequeño"
    elif area <= p66:
        label = "Mediano"
    else:
        label = "Grande"
    return label, area

def estimar_tamano(imagen: Image.Image, modelo_dict):
    t = modelo_dict["tipo"]
    if t in ("base", "base_cnn"):
        return estimar_tamano_base(imagen)
    return estimar_tamano_mejorado(imagen, modelo_dict["size_percentiles"])


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predecir(imagen_procesada, modelo_dict):
    modelo = modelo_dict["modelo"]
    encoder = modelo_dict["encoder"]
    if modelo_dict["arquitectura"] == "cnn":
        pred = modelo.predict(imagen_procesada, verbose=0)[0]
    else:
        pred = modelo.predict_proba(imagen_procesada)[0]
    clase_id = int(np.argmax(pred))
    confianza = float(pred[clase_id])
    clase = encoder.inverse_transform([clase_id])[0]
    probabilidades = {c: float(pred[i]) for i, c in enumerate(encoder.classes_)}
    return {
        "clase": clase,
        "confianza": confianza,
        "probabilidades": probabilidades,
    }


# ---------------------------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------------------------
st.title("🍎 Clasificador de Calidad de Frutas")
st.markdown(
    "Sube una imagen o captúrala con tu cámara para que el modelo evalúe "
    "su calidad en segundos."
)

modelo_dict = cargar_modelo()

tab1, tab2 = st.tabs(["📁 Subir imagen", "📷 Cámara"])

imagen = None

with tab1:
    archivo = st.file_uploader(
        "Selecciona una imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if archivo:
        imagen = Image.open(archivo).convert("RGB")
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

with tab2:
    cam = st.camera_input("Toma una foto", label_visibility="collapsed")
    if cam:
        imagen = Image.open(cam).convert("RGB")

if imagen is not None:
    tam_estimado, medida_tam = estimar_tamano(imagen, modelo_dict)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🔍 Clasificar", type="primary", use_container_width=True):
            with st.spinner("Clasificando imagen..."):
                img_proc = preprocesar_imagen(imagen, modelo_dict)
                resultado = predecir(img_proc, modelo_dict)

            st.markdown("---")
            st.subheader("📋 Resultados")

            clases_colors = {
                "bueno": ("🟢", "#4CAF50", "Fruta apta para consumo, sin defectos visibles."),
                "regular": ("🟡", "#FFB300", "Defectos menores. Puede requerir inspección adicional."),
                "malo": ("🔴", "#E53935", "Deterioro evidente. No apta para comercialización."),
            }

            clase = resultado["clase"]
            icono, color, desc = clases_colors.get(clase, ("⚪", "#999", ""))

            st.markdown(
                f"""
                <div style="
                    border: 2px solid {color}; border-radius: 12px;
                    padding: 1.25rem; text-align: center; margin-bottom: 1rem;
                ">
                    <div style="font-size: 2.5rem;">{icono}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {color};">
                        {clase.upper()}
                    </div>
                    <div style="font-size: 0.85rem; color: #aaa; margin-top: 0.25rem;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Confianza", f"{resultado['confianza']:.1%}")
            with col_b:
                st.metric("Tamaño estimado", tam_estimado)
            with col_c:
                if modelo_dict["tipo"] in ("base", "base_cnn"):
                    st.metric("Área de fruta", f"{medida_tam:.1%}")
                else:
                    st.metric("Área de fruta", f"{medida_tam:.0f} px")

            st.markdown("#### Probabilidades por clase")
            for cls_name, prob in resultado["probabilidades"].items():
                color_bar = clases_colors.get(cls_name, ("", "#ccc", ""))[1]
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.35rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <span>{cls_name.capitalize()}</span>
                            <span>{prob:.1%}</span>
                        </div>
                        <div style="
                            background: #262730; border-radius: 6px; height: 10px; width: 100%;
                        ">
                            <div style="
                                background: {color_bar}; width: {prob*100}%;
                                height: 10px; border-radius: 6px; transition: width 0.4s ease;
                            "></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.success("✅ Clasificación completada")
else:
    st.info("👆 Sube una imagen o usa la cámara para comenzar.")

st.markdown("---")
st.caption(
    "Proyecto Final — APO III — Universidad ICESI | "
    "Fase 6: Despliegue (CRISP-DM)"
)
