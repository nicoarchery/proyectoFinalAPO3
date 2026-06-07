import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import sys
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

st.set_page_config(
    page_title="Clasificación de Calidad de Frutas",
    page_icon="🍎",
    layout="centered",
)

# ===========================================================================
# SELECCION DE MODELO
# ===========================================================================
# Descomenta SOLO UNA para elegir el modelo:
#   "cnn"  - CNN (64x64) ← recomendado, mejor accuracy
#   "rf"   - Random Forest (64x64, features manuales)

#MODEL_TYPE = "rf"
MODEL_TYPE = "cnn"

# ===========================================================================
# MODEL LOADING
# ===========================================================================
@st.cache_resource
def cargar_modelo():
    cfg = joblib.load(OUTPUT_DIR / "feature_config.pkl")

    if MODEL_TYPE == "rf":
        return {
            "modelo": joblib.load(OUTPUT_DIR / "best_traditional_model_random_forest.pkl"),
            "encoder": joblib.load(OUTPUT_DIR / "label_encoder.pkl"),
            "config": cfg,
            "arquitectura": "tradicional",
        }

    if MODEL_TYPE == "cnn":
        from tensorflow import keras
        return {
            #"modelo": keras.models.load_model(OUTPUT_DIR / "cnn_model_mejorado.keras"),
            "modelo": keras.models.load_model(OUTPUT_DIR / "cnn_model.keras"),
            "encoder": joblib.load(OUTPUT_DIR / "label_encoder.pkl"),
            "config": cfg,
            "arquitectura": "cnn",
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

def preprocesar_tradicional(imagen: Image.Image):
    TARGET_SIZE = (64, 64)
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

def preprocesar_cnn(imagen: Image.Image):
    
    #arr = np.asarray(imagen.resize((96, 96)), dtype=np.float32) / 255.0
    arr = np.asarray(imagen.resize((64, 64)), dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocesar_imagen(imagen: Image.Image, modelo_dict):
    if modelo_dict["arquitectura"] == "cnn":
        return preprocesar_cnn(imagen)
    return preprocesar_tradicional(imagen)


# ---------------------------------------------------------------------------
# SIZE ESTIMATION
# ---------------------------------------------------------------------------
def estimar_tamano(imagen: Image.Image):
    img_cv = cv2.cvtColor(np.asarray(imagen.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    h_img, w_img = gray.shape
    cobertura = (w * h) / (w_img * h_img)

    return cobertura


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
    medida_tam = estimar_tamano(imagen)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        clicked = st.button("🔍 Clasificar", type="primary", use_container_width=True)

    if clicked:
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

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Confianza", f"{resultado['confianza']:.1%}")
        with col_b:
            st.metric("Cobertura en imagen", f"{medida_tam:.1%}")

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
