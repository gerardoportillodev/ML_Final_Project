# Unsupervised Learning Final Project

An advanced unsupervised learning analysis project implementing dimensionality reduction and clustering techniques on real-world multidimensional data.

## 📋 Project Overview

This project demonstrates comprehensive unsupervised machine learning techniques including:
- **Dimensionality Reduction**: PCA and UMAP
- **Clustering**: KMeans and DBSCAN
- **Evaluation**: Internal metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Interpretation**: Domain-specific insights and actionable recommendations

## 🎯 Project Goals

1. Apply and compare multiple dimensionality reduction techniques
2. Implement and evaluate various clustering algorithms
3. Extract meaningful patterns and insights from unlabeled data
4. Provide domain-specific interpretation of discovered segments
5. Generate actionable recommendations based on cluster analysis

## 📊 Dataset Description

**[TO BE FILLED BY USER]**

- **Source**: [Describe the data source - business/government/health/environment]
- **Size**: ≥1000 rows, ≥10 variables
- **Domain**: [Application domain]
- **Variables**: [Brief description of key variables]
- **Time Period**: [When was the data collected]

Place your dataset file as `data/raw/base_historica.csv` before running the analysis.

## 🗂️ Project Structure

```
ML_Final_Project/
├── data/
│   ├── raw/                      # Raw data files (base_historica.csv)
│   └── processed/                # Processed data files
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb           # EDA and data preparation
│   ├── 02_dimensionality_reduction.ipynb    # PCA and UMAP analysis
│   ├── 03_clustering.ipynb                  # KMeans and DBSCAN
│   └── 04_interpretation_conclusions.ipynb  # Insights and conclusions
├── src/
│   ├── data_loading.py           # Data loading utilities
│   ├── preprocessing.py          # Data cleaning and preprocessing
│   ├── dim_reduction.py          # Dimensionality reduction (PCA, UMAP)
│   ├── clustering.py             # Clustering algorithms
│   ├── evaluation.py             # Clustering evaluation metrics
│   └── __init__.py
├── reports/
│   ├── technical_report_template.md  # Detailed technical report
│   └── slides_outline.md             # Executive presentation outline
├── models/                       # Saved models and artifacts
├── logs/                         # Execution logs
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
└── .gitignore

```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/gerardoportillodev/ML_Final_Project.git
cd ML_Final_Project

# 2. Create virtual environment
python3 -m venv venv

# 3. Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Place your data file
# Copy your dataset to: data/raw/base_historica.csv
```

### Using Setup Scripts

**Linux/Mac**:
```bash
chmod +x setup.sh
./setup.sh
```

**Windows**:
```cmd
setup.bat
```

## 📓 Notebooks Workflow

### 1. EDA and Preprocessing (`01_eda_preprocessing.ipynb`)
- Dataset description and origin documentation
- Data loading and initial exploration
- Missing value analysis and treatment
- Outlier detection and handling
- Feature distributions and correlations
- Data cleaning and preparation

### 2. Dimensionality Reduction (`02_dimensionality_reduction.ipynb`)
- **PCA Implementation**:
  - Explained variance analysis
  - Component selection
  - Feature loadings interpretation
  - 2D and 3D visualizations
  
- **UMAP Implementation**:
  - Parameter tuning (n_neighbors, min_dist)
  - 2D and 3D projections
  - Comparison with PCA
  
- **Comparison**: Which method reveals better structure and why?

### 3. Clustering (`03_clustering.ipynb`)
- **KMeans**:
  - Elbow method for optimal k
  - Silhouette analysis
  - Cluster assignment and visualization
  
- **DBSCAN**:
  - Parameter selection (eps, min_samples)
  - Density-based cluster discovery
  - Noise point identification
  
- **Evaluation**:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Method comparison

### 4. Interpretation (`04_interpretation_conclusions.ipynb`)
- Cluster profiling (statistical characteristics)
- Domain-specific interpretation
- Actionable insights extraction
- Limitations discussion
- Future work recommendations

## 🔬 Evaluation Metrics

The project uses three internal clustering metrics implemented in `src/evaluation.py`:

1. **Silhouette Score** [-1, 1]: Measures cluster cohesion and separation (higher is better)
2. **Davies-Bouldin Index** [0, ∞): Measures average similarity ratio of clusters (lower is better)
3. **Calinski-Harabasz Index** [0, ∞): Variance ratio criterion (higher is better)

## 📈 Usage

### Running the Analysis

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Launch Jupyter
jupyter notebook

# Open and run notebooks in sequence:
# 1. notebooks/01_eda_preprocessing.ipynb
# 2. notebooks/02_dimensionality_reduction.ipynb
# 3. notebooks/03_clustering.ipynb
# 4. notebooks/04_interpretation_conclusions.ipynb
```

### Using the Modules Programmatically

```python
from src.preprocessing import DataPreprocessor
from src.dim_reduction import DimensionalityReducer
from src.clustering import ClusteringAnalyzer
from src.evaluation import ClusteringEvaluator

# Preprocess data
preprocessor = DataPreprocessor(scaling_method='standard')
data_clean = preprocessor.preprocess(data)

# Apply dimensionality reduction
reducer = DimensionalityReducer()
X_pca, pca_model = reducer.apply_pca(data_clean, variance_threshold=0.95)
X_umap, umap_model = reducer.apply_umap(data_clean, n_components=2)

# Cluster the data
clusterer = ClusteringAnalyzer()
labels_kmeans, kmeans_model = clusterer.apply_kmeans(X_pca, n_clusters=3)
labels_dbscan, dbscan_model = clusterer.apply_dbscan(X_pca, eps=0.5)

# Evaluate clustering
evaluator = ClusteringEvaluator()
metrics_kmeans = evaluator.evaluate_clustering(X_pca, labels_kmeans, 'kmeans')
metrics_dbscan = evaluator.evaluate_clustering(X_pca, labels_dbscan, 'dbscan')
```

## 📝 Reports

### Technical Report
A comprehensive technical report template is provided in `reports/technical_report_template.md` with sections for:
- Introduction and objectives
- Dataset description
- Methodology
- EDA findings
- Dimensionality reduction results
- Clustering results
- Domain interpretation
- Conclusions and future work
- References

### Presentation Slides
An executive presentation outline (≤15 slides) is provided in `reports/slides_outline.md` focusing on:
- Key findings
- Business value
- Actionable recommendations
- Visual summaries

## 🗺️ Rubric Requirements Mapping

| Requirement | Location in Repository |
|------------|------------------------|
| **Dataset (≥1000 rows, ≥10 vars)** | `data/raw/base_historica.csv` |
| **EDA & Preprocessing** | `notebooks/01_eda_preprocessing.ipynb` |
| **PCA Implementation** | `notebooks/02_dimensionality_reduction.ipynb`, `src/dim_reduction.py` |
| **UMAP Implementation** | `notebooks/02_dimensionality_reduction.ipynb`, `src/dim_reduction.py` |
| **KMeans Clustering** | `notebooks/03_clustering.ipynb`, `src/clustering.py` |
| **DBSCAN Clustering** | `notebooks/03_clustering.ipynb`, `src/clustering.py` |
| **Silhouette Score** | `src/evaluation.py` (line ~30) |
| **Davies-Bouldin Index** | `src/evaluation.py` (line ~60) |
| **Calinski-Harabasz Index** | `src/evaluation.py` (line ~90) |
| **Method Comparison** | `notebooks/03_clustering.ipynb`, `src/evaluation.py` |
| **2D/3D Visualizations** | All notebooks, visualization functions in `src/` |
| **Domain Interpretation** | `notebooks/04_interpretation_conclusions.ipynb` |
| **Technical Report** | `reports/technical_report_template.md` |
| **Presentation Slides** | `reports/slides_outline.md` |
| **Source Code** | `src/` directory |
| **Documentation** | `README.md` (this file), inline code comments |

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Preprocessing parameters
- Dimensionality reduction settings
- Clustering parameters
- Logging configuration

Example:
```yaml
dim_reduction:
  pca:
    variance_threshold: 0.95
  umap:
    n_components: 2
    n_neighbors: 15
    
clustering:
  kmeans:
    n_clusters: 3
  dbscan:
    eps: 0.5
    min_samples: 5
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 📦 Dependencies

Key libraries used:
- **Data Processing**: pandas, numpy
- **Dimensionality Reduction**: scikit-learn (PCA), umap-learn (UMAP)
- **Clustering**: scikit-learn (KMeans, DBSCAN), hdbscan
- **Visualization**: matplotlib, seaborn, plotly
- **Evaluation**: scikit-learn metrics
- **Notebooks**: jupyter, ipykernel

See `requirements.txt` for complete list.

## 🤝 Contributing

This is an academic project. If you have suggestions for improvements:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

ML Final Project Team - Unsupervised Learning Course

## 🙏 Acknowledgments

- Course instructors and teaching assistants
- scikit-learn, UMAP, and open-source ML community
- Dataset providers

## 📞 Support

For questions or issues:
- Open an issue in the GitHub repository
- Contact the project team
- Refer to the technical documentation in `reports/`

---

**Note**: This is an academic project for the Unsupervised Learning final assignment. The focus is on demonstrating mastery of unsupervised learning techniques, rigorous evaluation, and domain interpretation.
