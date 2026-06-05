import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
DATA_DIR      = "./data/chest_xray"
OUTPUT_DIR    = "./results/model"
BATCH_SIZE    = 32
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES       = ["NORMAL", "PNEUMONIA"]
# ────────────────────────────────────────────────────────


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def load_data(train_transform, val_transform):
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"), transform=val_transform)
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)} images")
    print(f"Test  : {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("./results/figures", exist_ok=True)

    print(f"Using device: {DEVICE}")

    train_transform, val_transform = get_transforms()
    train_loader, val_loader, test_loader = load_data(
        train_transform, val_transform)

    model     = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(OUTPUT_DIR, "best_model.pth"))
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"),
                   map_location=DEVICE))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    metrics = {
        "test_accuracy":    round(test_acc, 4),
        "best_val_accuracy": round(best_val_acc, 4)
    }
    with open("./results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTraining complete! Model saved to ./results/model/best_model.pth")


if __name__ == "__main__":
    main()