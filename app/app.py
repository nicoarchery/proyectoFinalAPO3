import streamlit as st
import numpy as np
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

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo_rf = joblib.load(OUTPUT_DIR / "best_traditional_model_random_forest.pkl")
    label_encoder = joblib.load(OUTPUT_DIR / "label_encoder.pkl")
    return {"rf": modelo_rf, "encoder": label_encoder}


# ---------------------------------------------------------------------------
# PREPROCESSING (placeholder)
# ---------------------------------------------------------------------------
def preprocesar_imagen(imagen: Image.Image, modelo_tipo: str = "rf"):
    """
    Preprocesa una imagen PIL para el modelo seleccionado.

    Para Random Forest (RF):
      1. Redimensionar con padding a 32x32
      2. Normalizar a [0, 1]
      3. Extraer: histograma de color (32 bins x 3 canales) + media (3) + std (3)
         → vector de 102 características

    Para CNN:
      1. Redimensionar con padding a 32x32
      2. Normalizar a [0, 1]
      3. Retornar array (1, 32, 32, 3)

    Parámetros:
        imagen: PIL Image en RGB.
        modelo_tipo: "rf" para Random Forest, "cnn" para CNN.

    Retorna:
        numpy array listo para el modelo.
    """
    TARGET_SIZE = (32, 32)
    HIST_BINS = 32

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

    img = resize_with_padding(imagen, TARGET_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    if modelo_tipo == "rf":
        features = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c].ravel(), bins=HIST_BINS, range=(0.0, 1.0))
            features.append(hist)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        return np.concatenate([np.concatenate(features), mean, std]).reshape(1, -1)

    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------
def predecir(imagen_procesada: np.ndarray, modelo, modelo_tipo: str = "rf"):
    proba = modelo["rf"].predict_proba(imagen_procesada)[0]
    clase_id = int(np.argmax(proba))
    confianza = float(proba[clase_id])
    clase = modelo["encoder"].inverse_transform([clase_id])[0]
    clases = list(modelo["encoder"].classes_)
    probabilidades = {c: float(proba[i]) for i, c in enumerate(clases)}
    return {
        "clase": clase,
        "confianza": confianza,
        "probabilidades": probabilidades,
    }


# ---------------------------------------------------------------------------
# ESTIMACIÓN DE TAMAÑO (simulación basada en área de píxeles no-fondo)
# ---------------------------------------------------------------------------
def estimar_tamano(imagen: Image.Image):
    """
    Estima el tamaño de la fruta basándose en el porcentaje de píxeles
    que no son del fondo blanco.
    """
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



# ---------------------------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------------------------
st.title("🍎 Clasificador de Calidad de Frutas")
st.markdown(
    "Sube una imagen o captúrala con tu cámara para que el modelo evalúe "
    "su calidad en segundos."
)

tab1, tab2 = st.tabs(["📁 Subir imagen", "📷 Cámara"])

imagen = None
fuente = None

with tab1:
    archivo = st.file_uploader(
        "Selecciona una imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if archivo:
        imagen = Image.open(archivo).convert("RGB")
        fuente = "archivo"
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

with tab2:
    cam = st.camera_input("Toma una foto", label_visibility="collapsed")
    if cam:
        imagen = Image.open(cam).convert("RGB")
        fuente = "cámara"

if imagen is not None:
    tam_estimado, ratio_pixeles = estimar_tamano(imagen)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if st.button("🔍 Clasificar", type="primary", use_container_width=True):
            with st.spinner("Clasificando imagen..."):
                modelo = cargar_modelo()
                img_proc = preprocesar_imagen(imagen, modelo_tipo="rf")
                resultado = predecir(img_proc, modelo, modelo_tipo="rf")

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
                st.metric(
                    "Área de fruta",
                    f"{ratio_pixeles:.1%}",
                )

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
