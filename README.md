# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project focused on detecting fraudulent transactions using imbalanced data. This project tracks the evolution from baseline linear models to Deep Learning architectures.

## 📌 Project Overview
Fraud detection is a classic imbalanced classification problem. This project documents the transition from simple statistical baselines to complex models, focusing on **Recall** and **Precision-Recall Tradeoffs** rather than raw accuracy.

## 📂 Project Structure
Managed with a modular MLOps-ready directory structure:
* `data/raw/`: Original dataset (ignored by Git for security).
* `data/processed/`: Scaled, split, and SMOTE-balanced data.
* `src/`: Python scripts for data processing, utility functions, and model training.

## 🚀 Phase 1: Baseline Logistic Regression
The first phase established a baseline using Vanilla Logistic Regression to understand the data's linear separability.

### The "Accuracy Trap"
While the model achieved an initial accuracy of **99.91%**, the Confusion Matrix revealed a critical flaw:
* **False Negatives (Missed Frauds):** 38
* **True Positives (Caught Frauds):** 49
In a banking context, missing 43% of fraudulent transactions is a massive security risk. This proved that **Accuracy** is a misleading metric for imbalanced datasets.

### Threshold Tuning
To improve the model, I analyzed the **Precision-Recall Curve** and adjusted the classification threshold to **0.05**.
* **Result:** Reduced False Negatives from **38 to 23**.
* **Tradeoff:** False Positives increased from **10 to 23**, a small price to pay for catching 15 more fraud cases.

## 🚀 Phase 2: Decision Trees & Complexity Analysis
Moving beyond linear boundaries, I implemented **Decision Trees** with `class_weight='balanced'` to capture non-linear fraud patterns.

### Depth vs. Performance Trade-off
I analyzed how model complexity impacts fraud detection using various depths:

| Configuration | False Negatives (FN) | False Positives (FP) | Insight |
| :--- | :---: | :---: | :--- |
| **max_depth=None** | **22** | **23** | Best "natural" balance; captures complexity but risks overfitting. |
| **max_depth=20** | **22** | **27** | Hit the limit of meaningful complexity; higher depth only added noise. |
| **max_depth=3** | **9** | **2916** | Hyper-sensitive; catches 90% of fraud but blocks too many customers. |

## 🚀 Phase 3: Data-Level Balancing (SMOTE)
Introduced **Synthetic Minority Over-sampling Technique (SMOTE)** to fix the imbalance at the data level rather than the model level.

* **Method:** Created synthetic fraud instances by interpolating between existing minority points using k-Nearest Neighbors.
* **Result:** For a Decision Tree (Depth 20), SMOTE pushed False Negatives down to **18**.
* **Key Learning:** While SMOTE helped the single tree, its impact on Ensembles (Random Forest) was negligible, proving that advanced algorithms often handle imbalance better than data-level hacks.

## 🚀 Phase 4: Ensemble Methods & GPU Acceleration
Transitioned to "Committee-based" learning using Random Forest and XGBoost, utilizing the **RTX 3050 GPU** for high-speed training.

### Performance Benchmarks
| Model | False Negatives (FN) | False Positives (FP) | Recall (Fraud ID %) |
| :--- | :---: | :---: | :--- |
| **Random Forest (No SMOTE)** | 20 | **5** | ~77.01% |
| **XGBoost (GPU Accelerated)** | **17** | 20 | **~80.46%** |

### The Dominance of V14
Using XGBoost's feature importance (Gain), I discovered that **V14** is the primary driver of fraud detection, providing over **12x more gain** than any other feature. This suggests that the latent space captured in V14 contains the most significant behavioral markers of fraudulent transactions.

## 🛠️ Technical Highlights
* **Modular Codebase:** Developed `confusion_matrix.py` to automate metric reporting and heatmap generation.
* **Hardware Acceleration:** Implemented `tree_method='hist'` and `device='cuda'` to leverage 3050 CUDA cores for training.
* **Cost-Sensitive Learning:** Utilized `scale_pos_weight` to mathematically penalize the model for missing the minority class.

## 🏗️ Upcoming Phases
- [ ] **Phase 5:** Anomaly detection using Deep Learning Autoencoders on **RTX 3050**.
- [ ] **Phase 6:** Model Deployment and API creation.