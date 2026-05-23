Proyecto Final
Algoritmos y Programación III
Semestre 2026-1
Facultad de Ingenierı́a, Diseño y Ciencias Aplicadas
Departamento de Computación y Sistemas Inteligentes
Ingenierı́a de Sistemas
Lineamientos para el proyecto final
El proyecto final del curso Algoritmos y Programación III es una actividad grupal (3
estudiantes por grupo) que busca desarrollar una solución a un problema real utilizando mo-
delos de analı́tica y conjuntos de datos de diversos formatos. Cada grupo deberá comprender
el problema, investigar su contexto y antecedentes, definir una metodologı́a de trabajo y
proponer métricas de desempeño adecuadas para evaluar el progreso. Se espera que, a lo
largo del desarrollo, se entrenen y evalúen diferentes modelos de analı́tica. Para cada modelo,
se deben ajustar adecuadamente los hiperparámetros y evaluar los resultados con base en
métricas predefinidas. Cada grupo deberá utilizar la metodologı́a CRISP-DM, adaptándola
a las necesidades particulares de su proyecto.
1.
Caso de estudio propuesto: sistema de anotación de
video
1.1.
 Contexto
En mercados y agroindustrias, la clasificación manual de productos frescos (tomates,
manzanas, papas, etc.) según su tamaño, madurez o defectos visibles es un proceso lento,
subjetivo y propenso a errores. Esta situación genera pérdidas económicas, desperdicio de
alimentos y falta de estandarización.
Se requiere desarrollar un sistema automático de clasificación de calidad basado en visión
por computadora, capaz de analizar imágenes de frutas o verduras, asignar una categorı́a
1
de calidad y estimar el tamaño relativo del producto. El sistema deberá ser entrenado con
un conjunto de datos que combine fuentes públicas y recolección propia, y será evaluado
mediante métricas rigurosas.
1.2.
 Entrada
Imágenes estáticas (fotografı́as tomadas con cámara web o celular) de una fruta o verdura
individual, sobre un fondo simple y uniforme.
1.3.
 Salidas
Clase de calidad (3 o más categorı́as definidas por grupo).
Estimación de tamaño (pequeño, mediano, grande) o diámetro en pı́xeles normalizados.
1.4.
 Formato de despliegue
Interfaz gráfica simple (por ejemplo, con Tkinter, PyQt o una aplicación web con Stream-
lit) que permita cargar una imagen o capturarla en tiempo real usando la cámara, y que
muestre la predicción obtenida.
1.5.
 Opcional (extensión)
Simulación de una lı́nea de empaque en tiempo real con cámara y Raspberry Pi.
2.
 Datos
2.1.
 Base de datos de referencia
Se utilizará el conjunto de datos Fruit Quality Classification, disponible en: https://
www.kaggle.com/datasets/ryandpark/fruit-quality-classification
Para la evaluación o el enriquecimiento de la base de datos, se puede emplear la carpe-
ta mixed quality, que contiene imágenes de diferentes frutas con distintas calidades. Los
estudiantes deberán segmentar individualmente cada fruta presente en las imágenes (si hay
varias por foto) para obtener ejemplares individuales.
2.2.
 Coordinación grupal y entre toda la clase
Cada grupo deberá recolectar al menos 30 a 50 imágenes adicionales de frutas o ver-
duras reales, obtenidas en plazas de mercado, supermercados o en sus propios hogares. Las
imágenes deben abarcar diferentes estados de madurez, tamaños y defectos (golpes, manchas,
podredumbre).
2
2.3.
 Anotación manual
Cada grupo debe etiquetar sus propias imágenes con la categorı́a de calidad y el tamaño
correspondiente. Las etiquetas deben ser coherentes con las definiciones establecidas para el
proyecto.
Entre todos los estudiantes se definirá un mecanismo para compartir anotaciones entre
todos los grupos (por ejemplo, una carpeta compartida en la nube o un repositorio común),
de modo que al final se disponga de un conjunto de datos más grande y diverso.
3.
 Materiales y métodos a utilizar
Los estudiantes deberán elegir al menos dos modelos distintos de machine learning
y uno de deep learning, entre los siguientes:
Modelos de machine learning tradicionales: Regresión logı́stica, K-vecinos más
cercanos (KNN), máquinas de soporte vectorial (SVM), Bagging, árboles de decisión,
Random Forest, XGBoost. Se deben ajustar hiperparámetros mediante validación cru-
zada o búsqueda en rejilla.
Redes neuronales convolucionales (CNN) simples: Arquitecturas pequeñas (por
ejemplo, 2 o 3 capas convolucionales + pooling + capas densas). Se permite el uso de
transfer learning siempre que se congelen capas y se añadan capas propias, aunque se
recomienda priorizar modelos entrenados desde cero para un mejor aprendizaje.
Redes neuronales totalmente conectadas (MLP) si se trabaja con caracterı́sticas
extraı́das manualmente.
4.
 Metodologı́a de trabajo: CRISP-DM
Cada grupo debe documentar explı́citamente cada fase de CRISP-DM adaptada a
su proyecto. No basta con copiar el diagrama; se debe mostrar evidencia de trabajo en cada
fase:

1. Comprensión del negocio: ¿Por qué es importante clasificar la calidad? ¿Qué im-
   pacto económico o social tiene?
2. Comprensión de los datos: Análisis exploratorio (distribución de clases, desbalanceo,
   calidad de imágenes, variabilidad).
3. Preparación de los datos: Procesamiento, manejo de clases desbalanceadas (si apli-
   ca).
4. Modelado: Entrenamiento, ajuste de hiperparámetros, selección del modelo.
5. Evaluación: Métricas en conjunto de prueba, análisis de errores, comparación con lı́nea
   base.
   3
6. Despliegue: Desarrollo de una interfaz gráfica sencilla que permita al usuario ver en
   tiempo real la calidad de una fruta presentada ante la cámara.
   Se debe entregar un diagrama de flujo personalizado que refleje la aplicación concreta
   de CRISP-DM.
7. 

 Evaluación
Se evaluará la calidad del trabajo mediante las siguientes preguntas:
¿La metodologı́a es clara y robusta?
¿Las aproximaciones realizadas en el proyecto son razonables?
¿Los datos se exploraron y procesaron de forma adecuada?
¿Las soluciones propuestas son ingeniosas e interesantes?
¿Se explican correctamente los impactos de la solución en el contexto abordado?
¿Se complementaron los datos iniciales?
¿Los estudiantes desarrollaron y transmitieron conocimientos no triviales sobre el pro-
blema, los algoritmos y los modelos?
¿El trabajo demuestra el desarrollo de las competencias definidas para este curso?
Se compartirá un documento para que un representante de cada grupo informe los in-
tegrantes y el enlace al repositorio de entrega en GitHub. El repositorio debe tener una
estructura clara, organizada según las fases definidas en la metodologı́a.
6.
 Aspectos a tener en cuenta
Se debe detallar el problema, la metodologı́a, las métricas para medir el progreso, los
datos recolectados, el análisis exploratorio de los datos y los siguientes pasos o etapas del
proyecto.
También se debe incluir un análisis de los aspectos éticos relevantes al implementar solu-
ciones de IA en el contexto del problema abordado.
Se debe detallar el entrenamiento de los modelos (incluyendo el ajuste de hiperparáme-
tros), los resultados obtenidos (métricas, gráficas, etc.) y el plan de despliegue. Además, se
realizará un análisis inicial de los impactos de la solución en el contexto del problema.
Se espera que el nivel de profundidad del análisis y la calidad de los resultados sean
rigurosos. Al final, se deberá presentar el análisis final de los impactos de la solución en el
contexto abordado.
Al concluir el proyecto, se debe incluir un video corto de no más de 10 minutos presentando
el proyecto, el contexto del problema, las técnicas utilizadas, los resultados y los principales
logros alcanzados.
4
Es fundamental que el código fuente esté bien documentado. Si se utilizan datos o código
fuente de terceros, deben referenciarse de forma clara y explı́cita; de lo contrario, se conside-
rará fraude.
Los informes deben contener explicaciones claras y concisas. Se recomienda incluir diagra-
mas de flujo, diagramas de bloques u otras figuras que ilustren la metodologı́a, la arquitectura
de software y los resultados, tanto en el informe como en la presentación. Se deben utilizar
gráficos de calidad vectorial.
7.
 Estructura básica del informe final
Máximo 7 páginas:

1. Tı́tulo.
2. Resumen (Abstract).
3. Introducción: contexto, descripción del problema, justificación de su interés.
4. Fundamentos teóricos: ¿Qué debe saber el lector para comprender el desarrollo? Nota:
   Seleccionar cuidadosamente el contenido de esta sección, evitando generalidades.
5. Metodologı́a: ¿Cómo se abordó el proyecto? Nota: No se busca una copia exacta del
   diagrama CRISP-DM.
6. Resultados: ¿Cómo se desempeñaron los modelos en diferentes conjuntos de datos?
   Métricas de interés para el problema especı́fico.
7. Análisis de resultados: ¿Qué se observa en los resultados? ¿Los modelos generalizan
   bien? ¿Hay sobreajuste (overfitting)? ¿Qué funciona bien? ¿Qué falla? ¿Cómo se com-
   paran los resultados con otros reportados en la literatura?
8. Conclusiones y trabajo futuro: ¿Qué se hizo? ¿Qué se aprendió? ¿Qué se puede mejorar?
9. Referencias bibliográficas: Incluir solo artı́culos, libros o materiales digitales que hayan
   sido leı́dos y utilizados. Usar formato IEEE.
   Observaciones
10. Revisar con detenimiento artı́culos publicados en conferencias de interés como NIPS,
    ICML o ICLR, entre otras, antes de redactar los informes finales.
11. TODOS los grupos deben prestar mucha atención a la rúbrica de evaluación, que estará
    presente en INTU, para ajustar su informe final adecuadamente.
    5
