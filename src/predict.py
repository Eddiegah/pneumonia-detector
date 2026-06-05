import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "./results/model/best_model.pth"
CLASSES    = ["NORMAL", "PNEUMONIA"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    return tensor, image


def predict(image_path, model):
    tensor, original_image = preprocess_image(image_path)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    predicted_idx   = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_idx]
    confidence      = float(probs[predicted_idx]) * 100

    return {
        "prediction":     predicted_class,
        "confidence":     round(confidence, 2),
        "probabilities": {
            "NORMAL":    round(float(probs[0]) * 100, 2),
            "PNEUMONIA": round(float(probs[1]) * 100, 2)
        },
        "original_image": original_image,
        "tensor":         tensor
    }