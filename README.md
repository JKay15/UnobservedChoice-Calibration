# Perfect Prediction of Unobservable Choice with an Imperfect Simulator

This repository contains the official implementation and experimental code for the paper **"Perfect Prediction of Unobservable Choice with an Imperfect Simulator"**.

We propose a framework to recover true unobserved choice probabilities (e.g., no-purchase) using only purchase data and a potentially biased simulator. We introduce two statistically consistent estimators:
1.  **Linear Calibration:** An exact recovery method via linear regression when the simulator bias is affine.
2.  **MRC Calibration:** A robust estimator based on Maximum Rank Correlation when the simulator bias is monotonic but non-linear.

## 📂 Project Structure

```text
.
├── data/                   # Place raw datasets (e.g., Expedia train.csv) here
├── results/                # Experiment logs and figures
│   ├── figures/            # Generated plots
│   └── logs/               # Raw CSV results
├── scripts/
│   ├── run_synthetic_experiment.py   # Main script for Synthetic Data (Exp 1-8)
│   ├── run_real_data_pipeline.py     # Main script for Real Data (Expedia)
├── src/
│   ├── algorithms/         # Core solvers (Linear, MRC, Multi-Sim)
│   ├── config.py           # Configuration classes
│   ├── datasets/           # Data loaders and Preprocessing logic
│   ├── engine/             # Synthetic data generation engine
│   ├── modules/            # Mappers for Z, U, Y (Linear/Neural)
│   └── utils/              # Metrics and Optimization helpers
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

* **Python 3.9+**
* **PyTorch (MPS/CUDA support recommended)**

### Installation

```bash
pip install -r requirements.txt
```

## 🧪 Experiments

### 1. Synthetic Data Experiments

**Reproduces the ablation studies (Consistency, Dimension Scaling, Robustness to Noise/Bias) and downstream Assortment Optimization regret analysis.**

```bash
python scripts/run_synthetic_experiment.py
```

* **Exp 1**: Convergence rate vs Sample Size (n).

* **Exp 2**: Calibration error vs Feature Dimension (d).

* **Exp 3**: Robustness to Utility Estimation Error (τ).

* **Exp 4**: Robustness of MRC across Linear vs. Monotone Biases.
* **Exp 5**: Downstream Assortment Revenue Regret.
* **Exp 6-8**: Multi-simulator robustness and Assortment size impacts.

### 2. Real Data Application (Expedia)

**Implements a semi-synthetic validation using the** [**Expedia Personalized Sort**](https://www.kaggle.com/competitions/expedia-personalized-sort/data) **dataset.**

* **Simulator**: Trained on historical **Click** **data (biased proxy).**
* **Calibration**: Performed on current **Booking** **data.**
* **Evaluation**: NLL on the full test set (including unobserved no-purchases).

**Steps:**

* **Download** **train.csv** **from Kaggle Expedia Challenge.**
* **Place it in** **data/train.csv**.
* **Run the pipeline:**

```bash
python scripts/run_real_data_pipeline.py
```

**This script compares:**

* **Uncalibrated Simulator** **(CatBoost)**
* **Linear Calibration** **(with Linear/Neural Utility)**
* **MRC Calibration** **(with Linear/Neural Utility)**

## 📊 Key Results

* **Theoretical Consistency**: Both estimators converge at the rate of $O(n^{1/2})$

* **High-Dimensional Scaling**: Error scales with $\sqrt{d}$, matching our theoretical bounds.

* **Robustness**: MRC significantly outperforms Linear Calibration when simulator bias is non-linear (Monotone).
* **Real World**: On Expedia data, Neural Utility + MRC Calibration reduces Negative Log Likelihood (NLL) from **1.72** **(Simulator) to** **0.45**, demonstrating significant practical value.
