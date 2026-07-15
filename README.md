# circuit-performance-shortcuts

Quantifying the Impact of Formula-Driven Dependencies in Circuit Performance Prediction
=======================================================================================

An end-to-end analytical pipeline designed to detect, measure, and statistically validate **data leakage** and **shortcut learning** in Machine Learning models trained on circuit performance datasets (such as GNN-derived circuit parameters).

When targets like the performance Figure of Merit ($FoM$) are deterministic mathematical combinations of physical attributes (such as bandwidth, phase margin, and gain), models can bypass physical domain learning. By simply reverse-engineering the target formula, they achieve artificially perfect scores. This repository provides the tools to measure exactly how your models perform when these formula-based "shortcuts" are removed.

📌 Executive Summary & Methodology
----------------------------------

Predicting analog circuit performance metrics using Graph Neural Networks (GNNs) or classic machine learning architectures often suffers from **formula-driven data leakage**. If an evaluation framework passes components of a formula directly to a model as input features, the model learns a proxy mathematical function instead of mapping physical properties.

### Mathematical Definition of Leakage

In a standard circuit performance prediction setting, we attempt to learn a mapping $f(X) \\to y$. However, if:

$$y = g(X\_{leak}) + \\epsilon$$

Where $X\_{leak} \\subset X$ and $g(\\cdot)$ is a known algebraic closed-form equation, models will exploit $X\_{leak}$ exclusively.

By running $10$-fold repeated train-test experiments, this pipeline isolates the shortcut components and quantifies the model performance gap.

### Key Pipeline Stages

1.  Real Dataset Analysis (perform101.csv): Analyzes correlations, fits a baseline linear estimator to evaluate $FoM$ linear dependencies, and tests model robustness across features.
    
2.  **Synthetic Dataset Generation:** Generates $10,000$ synthetic analog designs featuring realistic physical features (transistor counts, capacitance, parasitics) with structural noise and injected missing values.
    
3.  **Guardrail Preprocessing:** Robustly scales raw data, flags anomalous variances, performs median imputation on missing values, and processes highly skewed features using a specialized $\\log(1+x)$ scaling step.
    
4.  **Multi-Model Benchmark:** Evaluates Linear Regression, Random Forest, XGBoost, CatBoost, LightGBM, and an optimized PyTorch Multi-Layer Perceptron (MLP).
    
5.  **Statistical Verification:** Computes robust parametric and non-parametric indices to measure effect size:
    
    *   **Wilcoxon Signed-Rank Test** (for non-parametric paired significance)
        
    *   **Cohen's $d$ & Cliff's Delta** (for parametric and non-parametric effect scale)
        
    *   **Bootstrapped $95\\%$ Confidence Intervals** (to verify performance-drop boundaries)
        

🗂️ Directory Structure
-----------------------

Plaintext

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   ├── circuit_leakage_study.py           # Complete execution pipeline script  ├── perform101.csv                     # Real dataset (Optional: place here)  ├── statistical_comparison_results.csv # Saved output table from statistical tests  └── figures/                           # Auto-generated visualization output directory      ├── figure2_original_dataset_leakage.png          ├── figure3_synthetic_correlation_matrix.png      ├── figure4_synthetic_leakage_scatter.png      └── figure5_shortcut_comparison.png   `

## 🗂️ Directory Structure

```text
├── circuit_leakage_study.py           # Complete execution pipeline script
├── perform101.csv                     # Real dataset (Optional: place here)
├── statistical_comparison_results.csv # Saved output table from statistical tests
└── figures/                           # Auto-generated visualization output directory
    ├── figure2_original_dataset_leakage.png    
    ├── figure3_synthetic_correlation_matrix.png
    ├── figure4_synthetic_leakage_scatter.png
    └── figure5_shortcut_comparison.png
```

---

## 🛠️ Installation & Setup

**1. Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/circuit-performance-leakage.git
cd circuit-performance-leakage
```

**2. Set Up a Virtual Environment (Highly Recommended)**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install Requirements**
Install all baseline dependencies, including gradient boosting frameworks and PyTorch:
```bash
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn xgboost catboost lightgbm torch matplotlib seaborn
```
> **Note on PyTorch & GPU:** The pipeline automatically leverages CUDA if available; otherwise, it defaults cleanly to standard CPU processing.

---

## 🚀 Execution Guide

Run the complete validation suite by executing the main script:

```bash
python circuit_leakage_study.py
```

### Execution Steps & Logging
During runtime, the execution output will log step-by-step progress:
* Initializes random seeds and directory frameworks.
* Inspects `perform101.csv` if present to extract baseline formula coefficients.
* Synthesizes circuit records and validates the MCAR (Missing Completely at Random) assumption.
* Splits datasets into $10$ distinct configurations to perform repeated statistical modeling.
* Fits and evaluates all six learning frameworks across scenarios.
* Outputs a markdown-compatible summary table with statistical metrics.
* Populates the `figures/` directory with production-ready visualizations.

---

## 📊 Interpretive Metrics Guide

The terminal summary output generates several rigorous statistical indicators:

| Statistical Metric | Interpretation Guide |
| :--- | :--- |
| **`r2_with`** | Baseline $R^2$ score achieved when the shortcut feature is left in place. |
| **`r2_without`** | True $R^2$ performance of the model once the shortcut feature is removed. |
| **`abs_drop`** | Standard absolute difference; highly positive values show high leakage reliance. |
| **`cohens_d`** | Standardized difference of means; $> 0.8$ is Large, $> 1.2$ is Very Large degradation. |
| **`cliffs_delta`** | Robust, non-parametric ordinal measure of reliance. Range is $[-1.0, 1.0]$. |
| **`wilcoxon_p`** | Significance test; value $< 0.05$ confirms the performance change is statistically significant. |

---

## 📝 License

This project is open-source and licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).
