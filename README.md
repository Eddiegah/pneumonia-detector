
```markdown
# 🫁 Pneumonia Detection AI with Grad-CAM Explainability

> A deep learning system that detects pneumonia from chest X-rays and shows exactly where in the lung it found the abnormality — making the AI's decision visible and trustworthy.



![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)




![PyTorch](https://img.shields.io/badge/PyTorch-2.12-EE4C2C?style=flat-square&logo=pytorch)




![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=flat-square&logo=streamlit)




![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)



---

## 📌 Overview

Most medical AI models are black boxes — they give you a prediction with no explanation. This model uses **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate a heatmap showing exactly which region of the lung influenced the diagnosis. This is explainable AI applied to real clinical imaging.

The model is built on **ResNet18**, fine-tuned on 5,216 labeled chest X-rays, and wrapped in a full **Streamlit web UI** that anyone can use without writing a single line of code.

---

## 🎯 What It Does

- Takes any chest X-ray image as input
- Predicts **NORMAL** or **PNEUMONIA**
- Shows a **confidence score** on every prediction
- Generates a **Grad-CAM heatmap** highlighting the exact lung regions the model focused on
- Displays a **side-by-side view** of the original X-ray and the heatmap overlay
- Full **Streamlit web UI** — upload an image and get instant results

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **83.49%** |
| Best Validation Accuracy | **93.75%** |
| AUC-ROC | **0.9440** |
| Pneumonia Recall | **1.00** |
| Training Epochs | **10** |
| Dataset Size | **5,216 images** |

> Pneumonia recall of 1.00 means the model catches every single pneumonia case — critical in a clinical setting where false negatives are dangerous.

---

## 📈 Evaluation Plots

### Confusion Matrix
Shows the breakdown of correct and incorrect predictions across both classes.



![Confusion Matrix](results/figures/confusion_matrix.png)



### ROC Curve
AUC of 0.9440 demonstrates strong discriminative ability between normal and pneumonia cases.



![ROC Curve](results/figures/roc_curve.png)



---

## 🖥️ Live Demo

The project includes a fully interactive Streamlit web UI. Upload any chest X-ray and instantly see:
- The predicted diagnosis with confidence score
- The original X-ray and Grad-CAM heatmap side by side
- Full probability breakdown for both classes

```bash
streamlit run app.py
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Eddiegah/pneumonia-detector.git
cd pneumonia-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python src/train.py
```

### 5. Run evaluation and generate plots
```bash
python src/evaluate.py
```

### 6. Launch the web UI
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
pneumonia-detector/
├── app.py                  ← Streamlit web UI
├── requirements.txt        ← All dependencies
├── README.md
├── src/
│   ├── train.py            ← ResNet18 fine-tuning
│   ├── predict.py          ← Inference pipeline
│   ├── gradcam.py          ← Grad-CAM implementation
│   └── evaluate.py         ← Metrics and evaluation plots
└── results/
    ├── figures/
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    └── metrics.json
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning framework |
| ResNet18 | Pre-trained CNN backbone (ImageNet weights) |
| Grad-CAM | Explainability heatmap generation |
| Torchvision | Image transforms and model loading |
| Streamlit | Interactive web UI |
| scikit-learn | Evaluation metrics and ROC curve |
| OpenCV | Image processing and heatmap overlay |
| Matplotlib | Plot generation |
| Seaborn | Confusion matrix visualization |
| Pillow | Image handling |

---

## 🏗️ Architecture

```
Input X-Ray Image
        ↓
   Preprocessing
  (Resize 224x224, Normalize)
        ↓
ResNet18 (fine-tuned, last layer unfrozen)
        ↓
   Classification Head
  (Linear → 2 classes)
        ↓
  Prediction + Confidence
        ↓
   Grad-CAM Engine
  (Gradients from layer4)
        ↓
  Heatmap Overlay on X-Ray
```

---

## 💡 Why Grad-CAM?

Explainability is critical in medical AI. A model that says "pneumonia detected" without showing why cannot be trusted in a clinical setting. Grad-CAM bridges that gap by computing the gradient of the predicted class score with respect to the final convolutional feature maps, producing a localization map that highlights the regions most responsible for the prediction.

This means a radiologist can see exactly which part of the lung the AI flagged — making the system a tool for doctors rather than a replacement for them.

---

## 📦 Dataset

Chest X-Ray Images (Pneumonia) dataset with 5,216 labeled images split across train, validation, and test sets. Classes: NORMAL and PNEUMONIA.

---

## 🔗 Related Work

This project connects directly to my work at **Nexora**, a healthcare technology company focused on predictive modelling for early disease detection. The Grad-CAM explainability layer reflects a core principle of responsible medical AI — predictions must be interpretable to be clinically useful.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<p align="center">Built with 🫁 by <a href="https://github.com/Eddiegah">Eddiegah</a></p>
```
