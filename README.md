# 🫁 Pneumonia Detection AI with Grad-CAM Explainability

> An explainable deep learning system that detects pneumonia from chest X-ray images and visualizes exactly where the model found abnormalities using Grad-CAM.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

# 📌 Overview

Medical AI systems often act as **black boxes**, providing predictions without explaining how decisions are made. In healthcare, transparency is critical.

This project combines **Deep Learning** and **Explainable AI (XAI)** to detect pneumonia from chest X-ray images while showing clinicians exactly which regions influenced the diagnosis.

Built using a fine-tuned **ResNet18** architecture and enhanced with **Grad-CAM visualization**, the system provides both accurate predictions and interpretable explanations through an interactive **Streamlit web application**.

---

# 🎯 Key Features

✅ Binary classification of chest X-rays (**Normal vs Pneumonia**)

✅ Confidence score for every prediction

✅ Grad-CAM explainability heatmaps

✅ Side-by-side visualization of original image and model attention map

✅ Interactive Streamlit web interface

✅ Transfer learning with ResNet18

✅ Automated evaluation and performance reporting

---

# 🧠 Model Architecture

### Backbone Network
- ResNet18 (Pre-trained on ImageNet)
- Fine-tuned for binary pneumonia classification

### Explainability Layer
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Highlights lung regions contributing most to predictions

### Input
- Chest X-ray image

### Output
- Classification:
  - NORMAL
  - PNEUMONIA
- Prediction confidence
- Explainability heatmap

---

# 📊 Performance Metrics

| Metric | Score |
|----------|----------|
| Test Accuracy | **83.49%** |
| Best Validation Accuracy | **93.75%** |
| AUC-ROC | **0.9440** |
| Pneumonia Recall | **1.00** |
| Training Epochs | **10** |
| Dataset Size | **5,216 Images** |

### Clinical Significance

A pneumonia recall score of **1.00** means the model successfully identified every pneumonia case in the test set.

In medical screening systems, minimizing false negatives is often more important than maximizing overall accuracy, making recall a critical metric.

---

# 📈 Evaluation Results

## Confusion Matrix

![Confusion Matrix](results/figures/confusion_matrix.png)

---

## ROC Curve

![ROC Curve](results/figures/roc_curve.png)

---

# 🚀 Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Eddiegah/pneumonia-detector.git
cd pneumonia-detector
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the Model

```bash
python src/train.py
```

## 5. Launch the Web Application

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
pneumonia-detector/
│
├── app.py
├── requirements.txt
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── gradcam.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   └── roc_curve.png
│   │
│   └── metrics.json
│
└── README.md
```

---

# 🧰 Technology Stack

| Technology | Purpose |
|------------|----------|
| PyTorch | Deep Learning Framework |
| ResNet18 | Transfer Learning Backbone |
| Grad-CAM | Explainable AI Visualization |
| Streamlit | Interactive Web Application |
| OpenCV | Image Processing |
| scikit-learn | Model Evaluation |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |

---

# 💡 Why Explainable AI Matters

Traditional deep learning models can achieve high accuracy, but they rarely explain *why* a prediction was made.

For healthcare applications, trust and transparency are essential.

Grad-CAM helps bridge this gap by:

- Highlighting clinically relevant regions
- Improving model interpretability
- Supporting clinician decision-making
- Increasing confidence in AI-assisted diagnoses

This transforms the model from a simple classifier into an interpretable diagnostic support tool.

---

# 🔬 Future Improvements

- Multi-class lung disease classification
- Pneumonia severity estimation
- Model uncertainty quantification
- Attention-based architectures
- Deployment on cloud infrastructure
- Integration with electronic health systems

---

# 🏥 Research & Impact

This project demonstrates the practical application of **Explainable Artificial Intelligence (XAI)** in medical imaging.

The goal is not only to build accurate predictive models but also to create systems that clinicians can understand, trust, and confidently use in real-world healthcare environments.

---

# 🔗 About Nexora

This project aligns with the broader vision of **Nexora**, a healthcare technology initiative focused on:

- Predictive Healthcare
- Early Disease Detection
- Explainable AI
- Precision Medicine
- Clinical Decision Support Systems

---

# 📄 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

<div align="center">

### Built with ❤️, PyTorch, and Explainable AI

**Edmund Eric Gah (Eddiegah)**

[GitHub Profile](https://github.com/Eddiegah)

</div>
