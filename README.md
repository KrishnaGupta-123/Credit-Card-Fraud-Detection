# Credit Card Fraud Detection Pipeline

An end-to-end Machine Learning project focused on detecting fraudulent transactions using imbalanced data. This project tracks the evolution from baseline linear models to Deep Learning architectures.

## 📌 Project Overview
Fraud detection is a classic imbalanced classification problem. This project documents the transition from simple statistical baselines to complex models, focusing on **Recall** and **Precision-Recall Tradeoffs** rather than raw accuracy.

## 📂 Project Structure
Managed with a modular MLOps-ready directory structure:
* `data/raw/`: Original dataset (ignored by Git for security).
* `data/processed/`: Scaled and split data.
* `src/`: Python scripts for data processing and model training.

## 🚀 Phase 1: Baseline Logistic Regression
The first phase established a baseline using Vanilla Logistic Regression.

### The "Accuracy Trap"
While the model achieved an initial accuracy of **99.91%**, the Confusion Matrix revealed a critical flaw:
* **False Negatives (Missed Frauds):** 38
* **True Positives (Caught Frauds):** 49

In a banking context, missing 38 fraudulent transactions represents a high risk. This proved that **Accuracy** is a misleading metric for imbalanced datasets.

### Threshold Tuning
To improve the model, I analyzed the **Precision-Recall Curve** and adjusted the classification threshold to **0.05**.
* **Result:** Reduced False Negatives from **38 to 23**.
* **Tradeoff:** False Positives increased from **10 to 23**.

## 🚀 Phase 2: Decision Trees & Complexity Analysis
Moving beyond linear boundaries, I implemented **Decision Trees** with `class_weight='balanced'` to capture non-linear fraud patterns.

### Depth vs. Performance Trade-off
I analyzed how model complexity impacts fraud detection using various depths:

| Configuration | False Negatives (FN) | False Positives (FP) | Insight |
| :--- | :---: | :---: | :--- |
| **max_depth=None** | **22** | **23** | Best "natural" balance; highest risk of overfitting. |
| **max_depth=20** | **22** | **27** | Same recall as unconstrained, but increased false alarms. |
| **max_depth=3** | **9** | **2916** | Hyper-sensitive; catches almost all fraud but blocks too many customers. |

### Feature Importance & Selection
Using `.feature_importances_`, I identified that features like **V27** and **V17** contribute near-zero value to the classification. Dropping these irrelevant features helps simplify the model and reduce noise for future phases.

## 🛠️ Technical Highlights
* **Automated Hyperparameter Tuning:** Utilized `GridSearchCV` to explore optimal tree parameters.
* **Parallel Processing:** Implemented `n_jobs=-1` to utilize the multi-core architecture for faster training.

## 🏗️ Upcoming Phases
- [ ] **Phase 3:** Data-Level Balancing using **SMOTE** (Synthetic Minority Over-sampling Technique).
- [ ] **Phase 4:** Ensemble Methods (Random Forest / XGBoost).
- [ ] **Phase 5:** Anomaly detection using Deep Learning Autoencoders on **RTX 3050**.