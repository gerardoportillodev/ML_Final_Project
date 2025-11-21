# Análisis de Créditos Bancarios con Aprendizaje No Supervisado

Proyecto final de **Aprendizaje No Supervisado** aplicado a una base histórica de créditos bancarios.  
El objetivo es identificar **segmentos de riesgo** en la cartera mediante técnicas de **reducción de dimensionalidad** y **clustering**, sin usar etiquetas de default.

---

## 🎯 Objetivos del proyecto

1. Desarrollar un flujo completo de análisis no supervisado sobre un dataset real de créditos:
   - Limpieza y preprocesamiento de datos.
   - Reducción de dimensionalidad (PCA y UMAP).
   - Clustering (KMeans y DBSCAN).
   - Evaluación de calidad de los clusters con métricas internas.
   - Interpretación de segmentos en términos de riesgo crediticio.

2. Cumplir con la rúbrica de un **proyecto académico de aprendizaje no supervisado**, incluyendo:
   - Uso de al menos **dos técnicas de reducción de dimensionalidad**.
   - Uso de al menos **dos algoritmos de clustering**.
   - Evaluación mediante varias métricas internas (Silhouette, Davies-Bouldin, Calinski-Harabasz).
   - Entregables reproducibles (notebooks, datos procesados, resumen de resultados).

---

## 🗂️ Estructura del repositorio

Actualmente el repositorio está organizado así:

```bash
ML_FINAL_PROJECT/
├── notebooks/
│   ├── 1. Exploración y Preprocesamiento de Datos.ipynb
│   ├── 2. Reducción de Dimensionalidad.ipynb
│   ├── 3. Clustering.ipynb
│   └── 4. Interpretación y Conclusiones.ipynb
├── base_historica.csv          # Dataset original (créditos históricos)
├── data_processed.csv          # Dataset limpio y escalado, listo para modelar
├── data_with_clusters.csv      # Dataset con etiquetas de clusters (KMeans y DBSCAN)
├── embedding_pca_2d.csv        # Proyección 2D por PCA
├── embedding_umap_2d.csv       # Proyección 2D por UMAP
├── reportes/                   # Espacio para informe técnico y presentación
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

```

---

## 📓 Notebooks

### 1️⃣ `1. Exploración y Preprocesamiento de Datos.ipynb`

* Carga del dataset `base_historica.csv`.
* Análisis exploratorio:
  * Dimensiones, tipos de variables y primeras distribuciones.
  * Valores faltantes, duplicados y cardinalidad de variables categóricas.
* Tratamiento de valores faltantes:
  * Eliminación de columnas con más de 50% de nulos.
  * Imputación de numéricas (mediana) y categóricas (moda).
* Tratamiento de outliers:
  * Detección con IQR.
  * Winsorización en variables monetarias y de mora.
* Ingeniería de características de crédito:
  * Edad, antigüedad de la obligación, meses a vencimiento.
  * Ratios de riesgo (`ratio_mora`, `ratio_saldo_limite`).
  * Flags de mora (mora_30, etc.).
* Eliminación de PII y escalado de variables.
* Exporta: `data_processed.csv`.

---

### 2️⃣ `2. Reducción de Dimensionalidad.ipynb`

* Carga de `data_processed.csv`.
* Definición de la matriz de características `X`.
* **PCA**:
  * Cálculo de componentes principales.
  * Varianza explicada y varianza acumulada.
  * Scree plot para seleccionar número de componentes.
  * Proyección 2D (PC1 vs PC2).
* **UMAP**:
  * Cálculo de embedding 2D.
  * Visualización de la nube de puntos en 2D.
* Comparación cualitativa PCA vs UMAP como espacios para clustering.
* Exporta:
  * `embedding_pca_2d.csv`
  * `embedding_umap_2d.csv`.

---

### 3️⃣ `3. Clustering.ipynb`

* Carga de:
  * `data_processed.csv`
  * Proyecciones PCA/UMAP.
* **KMeans:**
  * Entrenamiento para k = 2,…,10.
  * Evaluación con:
    * Silhouette.
    * Davies-Bouldin.
    * Calinski-Harabasz.
  * Selección de modelos relevantes (por ejemplo k=2 y k=5).
  * Visualización de clusters en UMAP 2D.
* **DBSCAN:**
  * Búsqueda de parámetros (`eps`, `min_samples`).
  * Cálculo de métricas internas para configuraciones viables.
  * Identificación de clusters reales y puntos de ruido.
  * Visualización de clusters DBSCAN en UMAP 2D.
* Exporta: `data_with_clusters.csv` con columnas de clusters KMeans y DBSCAN.

---

### 4️⃣ `4. Interpretación y Conclusiones.ipynb`

* Carga de `data_with_clusters.csv`.
* Resumen de tamaños de cluster (KMeans y DBSCAN).
* Perfilamiento numérico de los clusters:
  * Medias por cluster de variables clave:
    * Días de mora, valor de mora, límites, saldos, cuotas.
    * Antigüedad, meses a vencimiento, ratios de riesgo.
    * Distribución de calificaciones de riesgo (A2, B, C, D, E).
  * Heatmaps de z-score por cluster (KMeans k=5 y DBSCAN).
* Interpretación de segmentos de clientes (clusters) en términos de:
  * Nivel de mora.
  * Intensidad del uso de límite.
  * Perfil de calificación de riesgo.
* Comparación entre KMeans y DBSCAN.
* Conclusiones, insights accionables y posibles líneas de trabajo futuro.

---

## 🔧 Requisitos e instalación

### Dependencias principales

Las dependencias se listan en `requirements.txt` e incluyen, entre otras:

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `umap-learn`
* `jupyter`

### Crear entorno virtual e instalar

```bash
# Clonar el repositorio
git clone https://github.com/gerardoportillodev/ML_FINAL_PROJECT.git
cd ML_FINAL_PROJECT

# Crear entorno virtual (ejemplo con venv)
python -m venv .venv

# Activar entorno
# macOS / Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar los notebooks

```bash
jupyter lab
# o
jupyter notebook
```

Orden sugerido:

1. `notebooks/1. Exploración y Preprocesamiento de Datos.ipynb`
2. `notebooks/2. Reducción de Dimensionalidad.ipynb`
3. `notebooks/3. Clustering.ipynb`
4. `notebooks/4. Interpretación y Conclusiones.ipynb`

---

## 📚 Relación con la rúbrica del curso

Este proyecto cumple con los puntos clave del **proyecto final de aprendizaje no supervisado**:

* ✅ Dataset real de créditos bancarios con múltiples variables.
* ✅ Dos técnicas de reducción de dimensionalidad: **PCA** y **UMAP**.
* ✅ Dos algoritmos de clustering: **KMeans** y **DBSCAN**.
* ✅ Evaluación con métricas internas:
  * Silhouette
  * Davies-Bouldin
  * Calinski-Harabasz
* ✅ Flujo completo documentado en cuatro notebooks.
* ✅ Segmentos interpretados en el contexto de **riesgo de crédito** y comportamiento de mora.

La carpeta `reportes/` está pensada para incluir:

* Informe técnico en PDF.
* Presentación ejecutiva (máx. 15 diapositivas) con los hallazgos clave.

---

## 📄 Licencia

Este proyecto se distribuye bajo la licencia indicada en el archivo `LICENSE`.

---

> Proyecto desarrollado con fines académicos. No constituye asesoría financiera ni reemplaza los modelos formales de riesgo de una institución bancaria.
