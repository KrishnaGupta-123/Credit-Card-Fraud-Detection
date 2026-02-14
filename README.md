# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project focused on detecting fraudulent transactions using imbalanced data. This project tracks the evolution from baseline linear models to optimized ensemble architectures.

## 📌 Project Overview
Fraud detection is a classic imbalanced classification problem. This project documents the transition from simple statistical baselines to complex models, focusing on **Recall** and **Precision-Recall Tradeoffs** rather than raw accuracy.

## 📂 Project Structure
Managed with a modular MLOps-ready directory structure:
* `data/raw/`: Original dataset (ignored by Git for security).
* `data/processed/`: Scaled, split, and SMOTE-balanced data.
* `src/`: Python scripts for data processing, utility functions, and model training.
* `models/`: Saved `.pkl` files of the best performing models.

## 🚀 Phase 1: Baseline Logistic Regression
Established a baseline using Vanilla Logistic Regression to understand the data's linear separability.

### The "Accuracy Trap"
While the model achieved an initial accuracy of **99.91%**, the Confusion Matrix revealed a critical flaw:
* **False Negatives (Missed Frauds):** 38
* **True Positives (Caught Frauds):** 49
In a banking context, missing 43% of fraudulent transactions is a massive security risk.

### Threshold Tuning
Adjusted the classification threshold to **0.05** based on the Precision-Recall Curve.
* **Result:** Reduced False Negatives from **38 to 23**.
* **Tradeoff:** False Positives increased from **10 to 23**.

## 🚀 Phase 2: Decision Trees & Complexity Analysis
Implemented **Decision Trees** with `class_weight='balanced'` to capture non-linear patterns.

| Configuration | False Negatives (FN) | False Positives (FP) | Insight |
| :--- | :---: | :---: | :--- |
| **max_depth=None** | **22** | **23** | Best "natural" balance. |
| **max_depth=3** | **9** | **2916** | Hyper-sensitive; too many false alarms. |

## 🚀 Phase 3: Data-Level Balancing (SMOTE)
Introduced **Synthetic Minority Over-sampling Technique (SMOTE)** to fix imbalance at the data level.
* **Method:** Created synthetic fraud instances using k-Nearest Neighbors interpolation.
* **Result:** For a Decision Tree (Depth 20), SMOTE pushed False Negatives down to **18**.

## 🚀 Phase 4: Ensemble Methods & GPU Acceleration
Transitioned to "Committee-based" learning using Random Forest and XGBoost, utilizing the **NVIDIA RTX 3050 GPU** for high-speed training and **Optuna** for Bayesian optimization.

### Performance Benchmarks
| Model | Configuration | FN (Security) | FP (Precision) | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Random Forest** | Baseline (No SMOTE) | 22 | **5** | **Precision King** |
| **Random Forest** | SMOTE Balanced | 19 | 16 | Balanced RF |
| **XGBoost** | GPU Baseline | 17 | 20 | High-Speed Base |
| **XGBoost** | **Optuna Optimized** | **16** | **33** | **Project Champion** |

### The Optuna Advantage
Implemented **Bayesian Optimization** to find the optimal Precision-Recall balance.
* **Penalty Function:** `Score = (FN * 25) + FP`.
* **Result:** Successfully matched the highest security (16 FN) while reducing False Positives by 38% compared to manual tuning.

### Feature Engineering & Importance
Using XGBoost's **Gain** metric, I analyzed the impact of engineered features:
* **Interaction Features:** Discovered that **V14_V4_interaction** is a primary driver of fraud detection.
* **The Dominance of V14:** Feature importance analysis revealed V14 provides over **12x more gain** than any other latent feature.

### 📊 Performance Benchmarks (Post-Feature Engineering & Optimization)
The following table tracks the evolution of the model's ability to detect fraud after implementing interaction features and pruning noise variables (V15, V22-V27).

| Model | Configuration | FN (Security) | FP (Precision) | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Logistic Regression** | Threshold default (No SMOTE) | 32 | 8 | **Linear Baseline** |
| **Logistic Regression** | Threshold 0.05 (Tuned) | 22 | 22 | **Recall Improved** |
| **Logistic Regression** | SMOTE Balanced | 10 | 1247 |**Hyper-Sensitive** |
| **Decision Tree** | Baseline (No SMOTE) | 22 | 28 | **Non-Linear Base** |
| **Decision Tree** | SMOTE Balanced | 19 | 253 | **High Sensitivity** |
| **Random Forest** | Baseline (No SMOTE) | 21 | **5** | **Precision King** |
| **Random Forest** | SMOTE Balanced | 18 | 11 | **Stable Ensemble** |
| **XGBoost** | GPU Baseline (CUDA) | 17 | 20 | **High-Speed Base** |
| **XGBoost** | **Optuna Optimized** | **16** | **33** | **Project Champion** |

> **Note on Feature Engineering Timing:** Although feature engineering was finalized in Phase 4, it was an iterative process. Pruning low-impact features and adding the **V14_V4 interaction** was the critical step that allowed the ensemble models to break the 18 FN barrier.

## 🛠️ Technical Highlights
* **Hardware Acceleration:** Leveraged 2,560 CUDA cores on the **RTX 3050** using `device='cuda'` for rapid 50-trial optimization studies.
* **Modular MLOps:** Developed automated scripts for metric reporting (`utils.py`) and model persistence (`joblib`).
* **Cost-Sensitive Learning:** Expertly balanced the tradeoff between security and customer friction using weighted penalty functions.

## 🏗️ Upcoming Phases
- [ ] **Phase 5:** Anomaly detection using Deep Learning Autoencoders on **RTX 3050**.
- [ ] **Phase 6:** Model Deployment and API creation.