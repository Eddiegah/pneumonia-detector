import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from tqdm import tqdm
from predict import load_model, DEVICE, CLASSES

# ── Config ──────────────────────────────────────────────
DATA_DIR    = "./data/chest_xray"
FIGURES_DIR = "./results/figures"
# ────────────────────────────────────────────────────────


def evaluate():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=transform)
    test_loader  = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("Loading model...")
    model = load_model()

    all_labels, all_preds, all_probs = [], [], []

    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images  = images.to(DEVICE)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            preds   = np.argmax(probs, axis=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1])

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASSES))

    # Confusion Matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Pneumonia Detector", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Pneumonia Detector", fontsize=13)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/roc_curve.png", dpi=150)
    plt.close()
    print("Saved: roc_curve.png")

    metrics = {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "auc_roc":  round(auc, 4)
    }
    with open("./results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll done! Check results/figures/ for your plots.")


if __name__ == "__main__":
    evaluate()