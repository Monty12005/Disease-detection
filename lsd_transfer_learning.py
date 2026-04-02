"""
Transfer Learning - Comparative Analysis of Pretrained Models for LSD Detection
Models: VGG16, ResNet50, InceptionV3
Reference: [8] Senthilkumar et al., "Early Detection of Lumpy Skin Disease in Cattle
           Using Deep Learning — A Comparative Analysis of Pretrained Models",
           Veterinary Sciences, 2024.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import copy
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"     : "dataset/",        # Root folder with train/ and val/ subfolders
    "num_classes"  : 2,                 # 0: Healthy, 1: LSD
    "batch_size"   : 32,
    "num_epochs"   : 20,
    "learning_rate": 0.001,
    "image_size"   : 224,               # InceptionV3 needs 299; handled per model
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir"     : "saved_models/",
    "class_names"  : ["Healthy", "LSD"],
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms(image_size=224):
    """Returns train and validation transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                             [0.229, 0.224, 0.225]),   # ImageNet std
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return {"train": train_transforms, "val": val_transforms}


def load_datasets(data_dir, image_size=224):
    """Load train and val datasets from folder structure."""
    tfms = get_transforms(image_size)
    image_datasets = {
        split: datasets.ImageFolder(
            root=os.path.join(data_dir, split),
            transform=tfms[split]
        )
        for split in ["train", "val"]
    }
    dataloaders = {
        split: DataLoader(
            image_datasets[split],
            batch_size=CONFIG["batch_size"],
            shuffle=(split == "train"),
            num_workers=4,
        )
        for split in ["train", "val"]
    }
    dataset_sizes = {s: len(image_datasets[s]) for s in ["train", "val"]}
    print(f"  Train: {dataset_sizes['train']} images")
    print(f"  Val  : {dataset_sizes['val']} images")
    return dataloaders, dataset_sizes


# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────
def build_vgg16(num_classes, freeze_features=True):
    """
    VGG16 with pretrained ImageNet weights.
    Replaces the final classifier layer for LSD binary classification.
    """
    model = models.vgg16(pretrained=True)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes, freeze_layers=True):
    """
    ResNet50 with pretrained ImageNet weights.
    Replaces the final fully connected layer.
    """
    model = models.resnet50(pretrained=True)

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # Unfreeze layer4 for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    return model


def build_inceptionv3(num_classes, freeze_layers=True):
    """
    InceptionV3 with pretrained ImageNet weights.
    Note: requires input size 299x299 and handles auxiliary output.
    """
    model = models.inception_v3(pretrained=True, aux_logits=True)

    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # Unfreeze last inception blocks
    for param in model.Mixed_7c.parameters():
        param.requires_grad = True

    # Replace primary classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Replace auxiliary classifier
    in_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(in_aux, num_classes)

    return model


MODEL_REGISTRY = {
    "VGG16"      : {"builder": build_vgg16,      "image_size": 224},
    "ResNet50"   : {"builder": build_resnet50,   "image_size": 224},
    "InceptionV3": {"builder": build_inceptionv3,"image_size": 299},
}


# ─────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────
def train_model(model, dataloaders, dataset_sizes, model_name, num_epochs=20):
    """Train and validate the model, returns best weights and history."""
    device     = CONFIG["device"]
    model      = model.to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    scheduler  = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0
    history        = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"  Device  : {device}")
    print(f"{'='*50}")

    for epoch in range(num_epochs):
        start = time.time()
        print(f"\nEpoch {epoch+1}/{num_epochs}  [{model_name}]")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss, running_corrects = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # InceptionV3 returns (output, aux_output) during training
                    if model_name == "InceptionV3" and phase == "train":
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss  = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss    = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss    / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"  {phase.upper():5s} — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc       = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        elapsed = time.time() - start
        print(f"  Epoch time: {elapsed:.1f}s")

    print(f"\n  Best Val Accuracy [{model_name}]: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)

    # Save best model
    save_path = os.path.join(CONFIG["save_dir"], f"{model_name}_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved → {save_path}")

    return model, history, best_acc.item()


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, dataloader, model_name):
    """Run inference on val set and print classification report."""
    device     = CONFIG["device"]
    model      = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"\n── Classification Report: {model_name} ──")
    print(classification_report(all_labels, all_preds,
                                target_names=CONFIG["class_names"]))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CONFIG["class_names"],
                yticklabels=CONFIG["class_names"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {model_name}_confusion_matrix.png")

    return all_preds, all_labels


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────
def plot_history(histories):
    """Plot training/validation accuracy and loss for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"VGG16": "#e74c3c", "ResNet50": "#2ecc71", "InceptionV3": "#3498db"}

    for name, history in histories.items():
        c = colors[name]
        axes[0].plot(history["train_acc"], linestyle="--", color=c, alpha=0.6, label=f"{name} Train")
        axes[0].plot(history["val_acc"],   linestyle="-",  color=c,            label=f"{name} Val")
        axes[1].plot(history["train_loss"], linestyle="--", color=c, alpha=0.6, label=f"{name} Train")
        axes[1].plot(history["val_loss"],   linestyle="-",  color=c,            label=f"{name} Val")

    for ax, title, ylabel in zip(axes,
                                  ["Model Accuracy Comparison", "Model Loss Comparison"],
                                  ["Accuracy", "Loss"]):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Transfer Learning — Pretrained Models Comparison for LSD Detection",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nComparison plot saved → comparison_plot.png")


def plot_accuracy_bar(results):
    """Bar chart comparing best val accuracy across models."""
    names = list(results.keys())
    accs  = [results[n] * 100 for n in names]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, accs, color=colors, edgecolor="black", linewidth=0.8)
    for bar_, acc in zip(bars, accs):
        plt.text(bar_.get_x() + bar_.get_width()/2,
                 bar_.get_height() + 0.5,
                 f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.ylim(0, 110)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Best Validation Accuracy — Model Comparison\n(LSD Detection)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.close()
    print("Accuracy bar chart saved → accuracy_comparison.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    results   = {}
    histories = {}

    for model_name, cfg in MODEL_REGISTRY.items():
        img_size   = cfg["image_size"]
        print(f"\n{'#'*55}")
        print(f"  MODEL: {model_name}  |  Input: {img_size}x{img_size}")
        print(f"{'#'*55}")

        # Load data with correct image size
        print("Loading dataset...")
        dataloaders, dataset_sizes = load_datasets(CONFIG["data_dir"], img_size)

        # Build model
        model = cfg["builder"](CONFIG["num_classes"])
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,}")

        # Train
        best_model, history, best_acc = train_model(
            model, dataloaders, dataset_sizes, model_name, CONFIG["num_epochs"]
        )

        # Evaluate
        evaluate_model(best_model, dataloaders["val"], model_name)

        results[model_name]   = best_acc
        histories[model_name] = history

    # Comparative plots
    plot_history(histories)
    plot_accuracy_bar(results)

    # Summary table
    print("\n" + "="*45)
    print(f"  {'Model':<14} {'Best Val Accuracy':>18}")
    print("="*45)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<14} {acc*100:>17.2f}%")
    best = max(results, key=results.get)
    print("="*45)
    print(f"  Best Model: {best} ({results[best]*100:.2f}%)")
    print("="*45)


if __name__ == "__main__":
    main()
