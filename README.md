# 🫁 Pneumonia Detection AI with Grad-CAM Explainability

> A deep learning system that detects pneumonia from chest X-rays and shows exactly where in the lung it found the abnormality.



![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)




![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch)




![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)



---

## 📌 Overview

Most medical AI models are black boxes — they give you a prediction with no explanation. This model uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate a heatmap showing exactly which region of the lung influenced the diagnosis. This is explainable AI applied to real clinical imaging.

---

## 🎯 What It Does

- Takes any chest X-ray image as input
- Predicts **NORMAL** or **PNEUMONIA**
- Shows **confidence score** on the prediction
- Generates a **Grad-CAM heatmap** highlighting the exact lung regions the model focused on
- Full **Streamlit web UI** — upload an image and get instant results

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **83.49%** |
| Best Val Accuracy | **93.75%** |
| AUC-ROC | **0.9440** |
| Pneumonia Recall | **1.00** |

Pneumonia recall of 1.00 means the model catches every pneumonia case — critical in a clinical context where false negatives are dangerous.

---

## 📈 Evaluation Plots

### Confusion Matrix


![Confusion Matrix](results/figures/confusion_matrix.png)



### ROC Curve


![ROC Curve](results/figures/roc_curve.png)



---

## 🚀 Quick Start

```bash
git clone https://github.com/Eddiegah/pneumonia-detector.git
cd pneumonia-detector
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run app.py
