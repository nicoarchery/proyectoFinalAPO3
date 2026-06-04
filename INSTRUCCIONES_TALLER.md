# Taller: Clasificación de Calidad de Frutas

## 📋 Descripción

Este taller implementa un sistema automático de clasificación de frutas en **tres categorías**:
- **Bueno**: Fruta sin defectos visibles, lista para venta
- **Regular**: Fruta con defectos leves (pequeños golpes, manchas)
- **Malo**: Fruta en mal estado (podredumbre, daño severo)

## 📂 Estructura de carpetas requerida

Crea la siguiente estructura de directorios en la raíz del proyecto:

```
proyectoFinalAPO3/
├── taller_calidad_frutas_bueno_regular_malo.ipynb
├── INSTRUCCIONES_TALLER.md
├── data/
│   ├── bueno/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.png
│   │   └── ...
│   ├── regular/
│   │   ├── imagen1.jpg
│   │   ├── imagen2.png
│   │   └── ...
│   └── malo/
│       ├── imagen1.jpg
│       ├── imagen2.png
│       └── ...
└── outputs/
    ├── best_traditional_model_*.pkl
    ├── label_encoder.pkl
    ├── feature_config.pkl
    └── cnn_model.keras (si TensorFlow está disponible)
```

## 🔧 Requisitos

### Dependencias automáticas
El notebook instala automáticamente los paquetes necesarios en la primera celda:
- **numpy**: Computación numérica
- **pandas**: Manipulación de datos
- **matplotlib** y **seaborn**: Visualización
- **scikit-learn**: Modelos clásicos (Logistic Regression, Random Forest, SVM)
- **pillow**: Procesamiento de imágenes
- **joblib**: Guardado de modelos

### Opcional
- **tensorflow/keras**: Para entrenar CNN (nota: Python 3.14 no soporta TensorFlow aún)

## 📝 Pasos de uso

### 1️⃣ Preparar los datos
1. Crea las carpetas `data/bueno`, `data/regular` y `data/malo`
2. Coloca **mínimo 30-50 imágenes por clase**:
   - Usa el dataset [Fruit Quality Classification](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification)
   - Añade imágenes reales tomadas en mercados o supermercados
   - Asegúrate de que cada imagen tenga fondo simple y uniforme

### 2️⃣ Abrir el notebook
En VS Code:
1. Abre `taller_calidad_frutas_bueno_regular_malo.ipynb`
2. Selecciona un kernel Python (se sugiere usar `.venv` o Anaconda)

### 3️⃣ Ejecutar celdas en orden

El notebook está dividido en **8 fases CRISP-DM**:

| Fase | Descripción | Acción |
|------|-------------|--------|
| 1 | **Importar librerías** | Ejecuta automáticamente |
| 2 | **Carga de datos** | Lee imágenes de `data/` |
| 3 | **Exploración visual** | Muestra ejemplos y distribución |
| 4 | **Preparación** | Divide en train/val/test |
| 5 | **Vectorización** | Extrae características de imágenes |
| 6 | **Entrenamiento** | Entrena 3 modelos (Logistic Regression, Random Forest, SVM) |
| 7 | **Evaluación** | Matriz de confusión, métricas F1, precision, recall |
| 8 | **Despliegue** | Guarda modelos en `outputs/` |

### 4️⃣ Interpretar resultados

Después de ejecutar todo, verás:
- **Gráficos de distribución** de clases
- **Comparación de modelos** en validación
- **Matriz de confusión** en test
- **Reporte de clasificación** (F1, Precision, Recall por clase)
- Ejemplos de imágenes **mal clasificadas**

### 5️⃣ Usar el modelo para predicciones

Al final del notebook hay una función `predict_image()`:

```python
prediction = predict_image("ruta/a/imagen.jpg")
print(f"Predicción: {prediction}")  # Output: bueno, regular o malo
```

## 📊 Modelos disponibles

### Modelos clásicos (entrenados)
1. **Logistic Regression** con features de píxeles e histogramas
2. **Random Forest** (100-200 árboles)
3. **SVM** lineal con escalado de features

→ El notebook selecciona automáticamente el **mejor modelo** en validación (maximizando F1 macro)

### Deep Learning (opcional)
- CNN simple (3 capas convolucionales) si TensorFlow está disponible
- ⚠️ **Nota**: Python 3.14 no soporta TensorFlow aún, se omite en ese caso

## 💾 Archivos generados

En `outputs/`:
- `best_traditional_model_*.pkl`: Modelo entrenado (e.g., `svm`, `log_reg`)
- `label_encoder.pkl`: Mapeador de etiquetas (bueno↔0, regular↔1, malo↔2)
- `feature_config.pkl`: Configuración de extracción de features

## 🎯 Recomendaciones

✅ **DO:**
- Usa imágenes de **buena calidad** (mínimo 200×200 px)
- Mantén **fondo uniforme** (preferiblemente blanco o gris)
- Asegúrate de **balance de clases** (similar # imágenes por categoría)
- Documenta **tus decisiones** de etiquetado

❌ **DON'T:**
- Usar imágenes muy pequeñas (<100 px)
- Mezclar diferentes tipos de frutas sin separar por tipo
- Olvidar de resaldar tus datos antes de entrenar

## 🤝 Próximos pasos

1. **Interfaz gráfica**: Crea una app con Tkinter/Streamlit para usar el modelo
2. **Mejora de datos**: Recolecta más imágenes para mejorar precisión
3. **Transferencia de aprendizaje**: Usa modelos pre-entrenados (ResNet, VGG) cuando TensorFlow esté disponible
4. **Aumento de datos**: Aplica rotaciones, cambios de iluminación, etc.

## ❓ FAQ

**P: ¿Qué pasa si no tengo imágenes aún?**
R: El notebook mostrará `Imágenes encontradas: 0` pero no fallará. Crea la estructura de carpetas y agrega imágenes.

**P: ¿Por qué TensorFlow no funciona?**
R: Python 3.14 es muy reciente. Baja a Python 3.11-3.13 para usar TensorFlow o usa solo modelos clásicos.

**P: ¿Cómo cambio el tamaño mínimo de imágenes?**
R: En la celda de vectorización, modifica `IMG_SIZE = (128, 128)` a tu valor deseado.

**P: ¿Puedo agregar más categorías?**
R: Sí. En la celda inicial cambia `CLASS_NAMES = ["bueno", "regular", "malo"]` a lo que necesites.

---

**Última actualización:** Mayo 2026  
**Autor(es):** Estudiantes de APO III - ICESI
