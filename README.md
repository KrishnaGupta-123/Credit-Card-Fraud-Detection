# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project focused on detecting fraudulent transactions using imbalanced data. This project tracks the evolution from baseline linear models to optimized ensemble architectures.

## 📌 Project Overview
Fraud detection is a classic imbalanced classification problem. This project documents the transition from simple statistical baselines to complex models, focusing on **Recall** and **Precision-Recall Tradeoffs** rather than raw accuracy.

## 📂 Project Structure
Managed with a modular MLOps-ready directory structure:
* `data/raw/`: Original dataset (ignored by Git for security).
* `data/processed/`: Scaled, split, and SMOTE-balanced data.
* `src/`: Python scripts for data processing, utility functions, and model training.
* `models/`: Saved `.pkl` and `.h5` files of the best performing models.

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


## 🚀 Phase 5: Deep Learning & Anomaly Detection
Transitioned to unsupervised and hybrid Deep Learning architectures using Autoencoders to capture complex, non-linear fraud signatures.

### Data Utilization & Input Refinement
To maintain continuity and performance, this phase utilized the optimized feature set developed in the previous stage:
* **Feature Engineered Inputs:** Leveraged the refined dataset containing the **V14_V4_interaction** and excluding noise features (V24, V26, V27, V22, V23, V15, V25), as this configuration yielded superior results in earlier ensemble trials.
* **Input Dimension:** The Autoencoder was built to process **24 high-signal features**, ensuring the model focused strictly on the most informative fraudulent patterns.

### Architecture & Optimization
Leveraged the **NVIDIA RTX 3050 GPU** to train a compressed latent representation of "Normal" transactions.
* **Structure:** Symmetric "Sandwich" architecture — Input (24) → 10 → 5 (Bottleneck) → 10 → Output (24).
* **Activations:** `tanh` for outer layers and `leaky_relu` for the bottleneck to prevent "dead neurons".
* **Training:** 150 epochs using Adam optimizer and MSE loss, focused on minimizing reconstruction error for legitimate data.

### 📊 Performance Benchmarks (Deep Learning)
By applying a **95th percentile threshold** on reconstruction error, the unsupervised model achieved the highest sensitivity in the project history.

| Model | Configuration | FN (Security) | FP (Precision) | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Autoencoder** | Unsupervised (150 Epochs) | **12** | 2,907 | **Recall Champion** |
| **Hybrid Model** | Softmax + Balanced Class | 16 | 2,918 | Hybrid Base |

### Key Insights
* **Unsupervised Breakthrough:** The Autoencoder identified 75 out of 87 frauds, surpassing the 16 FN of the Optuna-tuned XGBoost.
* **Softmax Integration:** Transitioned to supervised fine-tuning by attaching a Softmax head to the frozen encoder weights.
* **Balanced Class Performance:** Using balanced class weights with the Softmax head matched the XGBoost security benchmark with 16 False Negatives.
* **Model Comparison:** While the Deep Learning approach achieved the absolute lowest False Negatives (12 FN) in an unsupervised state, the Optuna-optimized XGBoost model remains the best overall performer. It provided the most professional balance between security and precision, maintaining 16 FN with only 33 False Positives compared to the high false-alarm rates of the Deep Learning trials.

### 🛠️ Technical Highlights (Deep Learning)
* **Hardware Acceleration:** Leveraged 2,560 CUDA cores on the RTX 3050 for rapid 50-trial optimization and Deep Learning training.
* **Modular MLOps:** Developed automated scripts for metric reporting and model persistence (`joblib` for ML, `.h5` for DL).
* **Iterative Discovery:** Discovered the 12 FN "Champion" weights through extensive iterative training, utilizing model persistence to capture optimal reconstruction boundaries.

## 🏗️ Upcoming Phases
- [ ] **Phase 6:** Model Deployment and API creation.