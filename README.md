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

## 🛠️ Technical Highlights
* **Robust Pathing:** Implemented `os.path` logic for location-aware scripts.
* **Environment Isolation:** Managed via Conda virtual environments.
* **Modular Code:** Decoupled data splitting (`split_data.py`) from model training.

## 🏗️ Upcoming Phases
- [ ] **Phase 2:** Non-linear modeling with Decision Trees.
- [ ] **Phase 3:** Anomaly detection using Deep Learning Autoencoders.