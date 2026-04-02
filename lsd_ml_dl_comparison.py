"""
Machine Learning + Deep Learning Comparative Analysis for LSD Detection in Cattle
Reference: [6] Malik & Kaushik, "Lumpy Skin Disease Detection in Cattle:
           A Machine Learning and Deep Learning Approach"

Models Compared:
  MACHINE LEARNING : SVM, Random Forest, KNN, Logistic Regression, Gradient Boosting
  DEEP LEARNING    : Custom CNN, VGG16 (Transfer Learning)

Pipeline:
  Image → Preprocessing → Feature Extraction → ML Models
                       → CNN / Transfer Learning → DL Models
  → Compare Accuracy, Precision, Recall, F1-Score
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── PyTorch ──────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

# ── Sklearn (ML Models) ──────────────────────
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.decomposition import PCA

# ── Image Processing ─────────────────────────
from PIL import Image

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"     : "dataset/",         # train/ and val/ subfolders
    "num_classes"  : 2,
    "class_names"  : ["Healthy", "LSD"],
    "image_size"   : 224,
    "batch_size"   : 32,
    "num_epochs"   : 15,
    "learning_rate": 0.001,
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir"     : "saved_models/",
    "pca_components": 100,               # PCA dims for ML feature vectors
}
os.makedirs(CONFIG["save_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# 1. DATA LOADING & TRANSFORMS
# ─────────────────────────────────────────────
def get_transforms():
    train_tfm = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return {"train": train_tfm, "val": val_tfm}


def load_dataloaders():
    tfms = get_transforms()
    datasets_dict = {
        split: datasets.ImageFolder(
            root=os.path.join(CONFIG["data_dir"], split),
            transform=tfms[split]
        )
        for split in ["train", "val"]
    }
    loaders = {
        split: DataLoader(
            datasets_dict[split],
            batch_size=CONFIG["batch_size"],
            shuffle=(split == "train"),
            num_workers=2,
        )
        for split in ["train", "val"]
    }
    sizes = {s: len(datasets_dict[s]) for s in ["train", "val"]}
    print(f"  Train: {sizes['train']} | Val: {sizes['val']}")
    return loaders, sizes


# ─────────────────────────────────────────────
# 2. HANDCRAFTED FEATURE EXTRACTION (for ML)
# ─────────────────────────────────────────────
def extract_features_hog(image_tensor):
    """
    Extracts HOG-like features from a normalized image tensor.
    In practice, use skimage.feature.hog on raw PIL images.
    Here we use flattened + gradient approximation for portability.
    """
    img = image_tensor.numpy()                     # (3, H, W)
    gray = 0.299*img[0] + 0.587*img[1] + 0.114*img[2]  # grayscale

    # Gradient magnitude (simple Sobel-like)
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    magnitude = np.sqrt(gx**2 + gy**2)

    # Divide into 8x8 cells, compute histogram
    cell_size = 8
    h, w = magnitude.shape
    features = []
    for i in range(0, h - cell_size, cell_size):
        for j in range(0, w - cell_size, cell_size):
            cell = magnitude[i:i+cell_size, j:j+cell_size]
            hist, _ = np.histogram(cell, bins=9, range=(0, 1))
            features.extend(hist)

    return np.array(features, dtype=np.float32)


def extract_color_histogram(image_tensor, bins=32):
    """Per-channel color histogram features."""
    img = image_tensor.numpy()   # (3, H, W)
    features = []
    for c in range(3):
        hist, _ = np.histogram(img[c], bins=bins, range=(-2.5, 2.5))
        features.extend(hist / (hist.sum() + 1e-8))
    return np.array(features, dtype=np.float32)


def extract_ml_features(dataloader):
    """
    Combines HOG + Color Histogram features for all images.
    Returns feature matrix X and labels y.
    """
    print("  Extracting ML features (HOG + Color Histogram)...")
    X, y = [], []
    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            hog_feat   = extract_features_hog(img)
            color_feat = extract_color_histogram(img)
            combined   = np.concatenate([hog_feat, color_feat])
            X.append(combined)
            y.append(lbl.item())
    X = np.array(X)
    y = np.array(y)
    print(f"  Feature shape: {X.shape}")
    return X, y


# ─────────────────────────────────────────────
# 3. MACHINE LEARNING MODELS
# ─────────────────────────────────────────────
ML_MODELS = {
    "SVM (RBF)"          : SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    "Random Forest"      : RandomForestClassifier(n_estimators=200, max_depth=15,
                                                  random_state=42, n_jobs=-1),
    "KNN (k=5)"          : KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                              multi_class="auto"),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=150,
                                                       learning_rate=0.1,
                                                       max_depth=4, random_state=42),
}


def train_ml_models(X_train, y_train, X_val, y_val):
    """Train all ML models and return evaluation metrics."""
    print("\n" + "="*55)
    print("  MACHINE LEARNING MODELS")
    print("="*55)

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # PCA dimensionality reduction
    pca     = PCA(n_components=min(CONFIG["pca_components"], X_train.shape[1]))
    X_train = pca.fit_transform(X_train)
    X_val   = pca.transform(X_val)
    print(f"  PCA: {X_train.shape[1]} components retained")

    results = {}
    for name, clf in ML_MODELS.items():
        print(f"\n  Training: {name}")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)

        acc  = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, average="weighted", zero_division=0)
        rec  = recall_score(y_val, preds, average="weighted", zero_division=0)
        f1   = f1_score(y_val, preds, average="weighted", zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec,
                         "recall": rec, "f1": f1, "preds": preds}

        print(f"    Acc: {acc*100:.2f}%  Prec: {prec:.4f}  "
              f"Rec: {rec:.4f}  F1: {f1:.4f}")

    return results, y_val


# ─────────────────────────────────────────────
# 4. DEEP LEARNING — CUSTOM CNN
# ─────────────────────────────────────────────
class LSD_CNN(nn.Module):
    """
    Custom lightweight CNN for LSD binary classification.
    Architecture: 4 conv blocks + FC head
    """
    def __init__(self, num_classes=2):
        super(LSD_CNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 128),          nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# 5. DEEP LEARNING — VGG16 Transfer Learning
# ─────────────────────────────────────────────
def build_vgg16_transfer(num_classes):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    # Unfreeze last conv block
    for param in model.features[24:].parameters():
        param.requires_grad = True
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Linear(in_feat, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


# ─────────────────────────────────────────────
# 6. TRAINING LOOP (DL)
# ─────────────────────────────────────────────
def train_dl_model(model, loaders, sizes, model_name):
    device    = CONFIG["device"]
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc, best_wts = 0.0, None
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    print(f"\n  Training DL Model: {model_name}")
    for epoch in range(CONFIG["num_epochs"]):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct = 0.0, 0

            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels).item()

            epoch_loss = running_loss    / sizes[phase]
            epoch_acc  = running_correct / sizes[phase]
            history[f"{phase}_acc"].append(epoch_acc)
            history[f"{phase}_loss"].append(epoch_loss)

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc  = epoch_acc
                    best_wts  = model.state_dict().copy()

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:02d}/{CONFIG['num_epochs']}  "
                  f"Val Acc: {history['val_acc'][-1]*100:.2f}%  "
                  f"Val Loss: {history['val_loss'][-1]:.4f}")

    model.load_state_dict(best_wts)
    print(f"  Best Val Acc [{model_name}]: {best_acc*100:.2f}%")
    return model, history, best_acc


def evaluate_dl_model(model, loader, model_name):
    """Inference pass for DL model; returns metrics dict."""
    device = CONFIG["device"]
    model.eval().to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    print(f"\n── {model_name} ──")
    print(classification_report(all_labels, all_preds,
                                target_names=CONFIG["class_names"]))
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "preds": all_preds}, all_labels


# ─────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=CONFIG["class_names"],
                yticklabels=CONFIG["class_names"])
    plt.title(title)
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout()
    fname = title.replace(" ", "_").replace("/", "_") + "_cm.png"
    plt.savefig(fname, dpi=150); plt.close()
    print(f"  Saved: {fname}")


def plot_metrics_comparison(all_results):
    """
    Side-by-side bar charts for Accuracy, Precision, Recall, F1
    across all ML and DL models.
    """
    names   = list(all_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x      = np.arange(len(names))
    width  = 0.2
    fig, ax = plt.subplots(figsize=(16, 6))

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [all_results[n][metric] * 100 for n in names]
        bars = ax.bar(x + i * width, vals, width, label=label,
                      color=color, edgecolor="white", linewidth=0.6)
        for bar_ in bars:
            h = bar_.get_height()
            ax.text(bar_.get_x() + bar_.get_width()/2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("LSD Detection — ML vs Deep Learning Model Comparison\n"
                 "(Accuracy / Precision / Recall / F1-Score)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Vertical separator between ML and DL models
    n_ml = len(ML_MODELS)
    ax.axvline(x=n_ml - 0.3, color="gray", linestyle="--", linewidth=1.2)
    ax.text(n_ml - 0.6, 108, "ML", fontsize=11, color="gray", ha="right")
    ax.text(n_ml - 0.1, 108, "DL", fontsize=11, color="gray", ha="left")

    plt.tight_layout()
    plt.savefig("ml_dl_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nComparison chart saved → ml_dl_comparison.png")


def plot_dl_training(histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    palette   = {"Custom CNN": "#e74c3c", "VGG16 (Transfer)": "#3498db"}

    for name, hist in histories.items():
        c = palette[name]
        axes[0].plot(hist["train_acc"], "--", color=c, alpha=0.6, label=f"{name} Train")
        axes[0].plot(hist["val_acc"],   "-",  color=c,            label=f"{name} Val")
        axes[1].plot(hist["train_loss"],"--", color=c, alpha=0.6, label=f"{name} Train")
        axes[1].plot(hist["val_loss"],  "-",  color=c,            label=f"{name} Val")

    for ax, title, ylabel in zip(axes,
                                  ["DL Accuracy", "DL Loss"],
                                  ["Accuracy", "Loss"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("Deep Learning Training Curves — LSD Detection",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("dl_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("DL training curves saved → dl_training_curves.png")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*55)
    print("  LSD Detection: ML + DL Comparative Analysis")
    print("  Malik & Kaushik [6]")
    print("█"*55)

    # ── Load data ───────────────────────────────
    print("\n[1] Loading Data...")
    loaders, sizes = load_dataloaders()

    all_results = {}
    dl_histories = {}

    # ── MACHINE LEARNING ────────────────────────
    print("\n[2] Feature Extraction for ML Models...")
    X_train, y_train = extract_ml_features(loaders["train"])
    X_val,   y_val   = extract_ml_features(loaders["val"])

    ml_results, y_true_ml = train_ml_models(X_train, y_train, X_val, y_val)
    all_results.update(ml_results)

    # Confusion matrices for ML
    for name, res in ml_results.items():
        plot_confusion_matrix(y_true_ml, res["preds"], name)

    # ── DEEP LEARNING ────────────────────────────
    print("\n" + "="*55)
    print("  DEEP LEARNING MODELS")
    print("="*55)

    dl_models = {
        "Custom CNN"      : LSD_CNN(CONFIG["num_classes"]),
        "VGG16 (Transfer)": build_vgg16_transfer(CONFIG["num_classes"]),
    }

    y_true_dl = None
    for name, model in dl_models.items():
        trained_model, history, _ = train_dl_model(model, loaders, sizes, name)
        dl_res, y_true_dl = evaluate_dl_model(trained_model, loaders["val"], name)
        all_results[name]   = dl_res
        dl_histories[name]  = history
        plot_confusion_matrix(y_true_dl, dl_res["preds"], name)

        save_path = os.path.join(CONFIG["save_dir"], f"{name.replace(' ', '_')}.pth")
        torch.save(trained_model.state_dict(), save_path)

    # ── PLOTS ────────────────────────────────────
    print("\n[3] Generating Comparison Plots...")
    plot_metrics_comparison(all_results)
    plot_dl_training(dl_histories)

    # ── SUMMARY TABLE ────────────────────────────
    print("\n" + "="*65)
    print(f"  {'Model':<25} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("="*65)
    for name, res in all_results.items():
        tag = " [DL]" if name in dl_models else " [ML]"
        print(f"  {name+tag:<30} "
              f"{res['accuracy']*100:>6.2f}%  "
              f"{res['precision']:>6.4f}  "
              f"{res['recall']:>6.4f}  "
              f"{res['f1']:>6.4f}")
    best = max(all_results, key=lambda k: all_results[k]["accuracy"])
    print("="*65)
    print(f"  Best Model: {best}  "
          f"({all_results[best]['accuracy']*100:.2f}% accuracy)")
    print("="*65)


if __name__ == "__main__":
    main()
