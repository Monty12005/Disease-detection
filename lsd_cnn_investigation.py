"""
Extensive Investigation of CNN Architectures for LSD Diagnosis in Dairy Cows
Reference: [7] D. K. Saha, "An Extensive Investigation of Convolutional Neural Network
           Designs for the Diagnosis of Lumpy Skin Disease in Dairy Cows",
           Heliyon, 2024.

CNN Architectures Investigated:
  1. LeNet-5          (Classic shallow CNN)
  2. AlexNet          (Deep CNN with large kernels)
  3. Custom VGG-style (3, 4, 5 conv-block variants)
  4. ResNet-style     (With residual skip connections)
  5. DenseNet-style   (With dense connections)
  6. Inception-style  (Multi-scale parallel convolutions)
  7. Lightweight CNN  (Depthwise separable convolutions)

Each architecture is trained, evaluated, and compared on:
  Accuracy, Precision, Recall, F1-Score, Parameters, Inference Time
"""

import os
import time
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              classification_report, confusion_matrix)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "data_dir"     : "dataset/",
    "num_classes"  : 2,
    "class_names"  : ["Healthy", "LSD"],
    "image_size"   : 224,
    "batch_size"   : 32,
    "num_epochs"   : 20,
    "learning_rate": 0.001,
    "weight_decay" : 1e-4,
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir"     : "saved_models/",
    "results_dir"  : "results/",
}
os.makedirs(CONFIG["save_dir"],    exist_ok=True)
os.makedirs(CONFIG["results_dir"], exist_ok=True)


# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────
def get_dataloaders():
    train_tfm = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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
    ds = {
        s: datasets.ImageFolder(os.path.join(CONFIG["data_dir"], s),
                                transform=(train_tfm if s == "train" else val_tfm))
        for s in ["train", "val"]
    }
    loaders = {
        s: DataLoader(ds[s], batch_size=CONFIG["batch_size"],
                      shuffle=(s == "train"), num_workers=2)
        for s in ["train", "val"]
    }
    sizes = {s: len(ds[s]) for s in ["train", "val"]}
    print(f"  Train: {sizes['train']} | Val: {sizes['val']}")
    return loaders, sizes


# ═════════════════════════════════════════════
# CNN ARCHITECTURE DEFINITIONS
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. LeNet-5  (Classic, shallow)
# ─────────────────────────────────────────────
class LeNet5(nn.Module):
    """
    Adapted LeNet-5 for 224x224 RGB input.
    Original designed for 32x32 grayscale; adjusted filters and FC dims.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),   nn.Tanh(), nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),  nn.Tanh(), nn.AvgPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 53 * 53, 120), nn.Tanh(),
            nn.Linear(120, 84),           nn.Tanh(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 2. AlexNet-style
# ─────────────────────────────────────────────
class AlexNetStyle(nn.Module):
    """
    AlexNet-inspired architecture adapted for LSD classification.
    Uses large kernels (11, 5) in early layers to capture coarse patterns.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  96, 11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256,  5, padding=2),           nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),           nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),           nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256*6*6, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024),    nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 3. VGG-style variants (3 / 4 / 5 blocks)
# ─────────────────────────────────────────────
def _vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1),
                   nn.BatchNorm2d(out_ch), nn.ReLU()]
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class VGGStyle(nn.Module):
    """
    Configurable VGG-style CNN.
    num_blocks: 3 → VGG-small, 4 → VGG-medium, 5 → VGG-large
    """
    def __init__(self, num_classes=2, num_blocks=4):
        super().__init__()
        configs = {
            3: [(3,  64,  2), (64,  128, 2), (128, 256, 3)],
            4: [(3,  64,  2), (64,  128, 2), (128, 256, 3), (256, 512, 3)],
            5: [(3,  64,  2), (64,  128, 2), (128, 256, 3),
                (256, 512, 3), (512, 512, 3)],
        }
        blocks = [_vgg_block(ic, oc, nc) for ic, oc, nc in configs[num_blocks]]
        self.features   = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(configs[num_blocks][-1][1] * 16, 1024),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 4. ResNet-style  (skip connections)
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels*stride, 3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels*stride), nn.ReLU(),
            nn.Conv2d(channels*stride, channels*stride, 3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels*stride),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(channels, channels*stride, 1, stride=stride, bias=False),
            nn.BatchNorm2d(channels*stride),
        ) if downsample else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class ResNetStyle(nn.Module):
    """
    Custom ResNet-style network with skip connections.
    Investigates how residual learning aids LSD feature extraction.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = nn.Sequential(ResidualBlock(64),  ResidualBlock(64))
        self.layer2 = nn.Sequential(ResidualBlock(64,  downsample=True),
                                    ResidualBlock(128))
        self.layer3 = nn.Sequential(ResidualBlock(128, downsample=True),
                                    ResidualBlock(256))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)


# ─────────────────────────────────────────────
# 5. DenseNet-style  (dense connections)
# ─────────────────────────────────────────────
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(),
            nn.Conv2d(in_ch, 4*growth_rate, 1, bias=False),
            nn.BatchNorm2d(4*growth_rate), nn.ReLU(),
            nn.Conv2d(4*growth_rate, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, num_layers, growth_rate):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.block   = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class DenseNetStyle(nn.Module):
    """
    DenseNet-style architecture: every layer receives feature maps from all
    preceding layers — maximises information flow for subtle LSD lesion features.
    """
    def __init__(self, num_classes=2, growth_rate=16):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        b1 = DenseBlock(64,  6, growth_rate)
        t1_ch = b1.out_channels // 2
        self.dense1 = b1
        self.trans1 = nn.Sequential(
            nn.BatchNorm2d(b1.out_channels), nn.ReLU(),
            nn.Conv2d(b1.out_channels, t1_ch, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )
        b2 = DenseBlock(t1_ch, 12, growth_rate)
        t2_ch = b2.out_channels // 2
        self.dense2 = b2
        self.trans2 = nn.Sequential(
            nn.BatchNorm2d(b2.out_channels), nn.ReLU(),
            nn.Conv2d(b2.out_channels, t2_ch, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )
        b3 = DenseBlock(t2_ch, 8, growth_rate)
        self.dense3 = b3
        self.head = nn.Sequential(
            nn.BatchNorm2d(b3.out_channels), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(b3.out_channels, num_classes),
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.trans1(self.dense1(x))
        x = self.trans2(self.dense2(x))
        x = self.dense3(x)
        return self.head(x)


# ─────────────────────────────────────────────
# 6. Inception-style  (multi-scale convolutions)
# ─────────────────────────────────────────────
class InceptionModule(nn.Module):
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, 1), nn.BatchNorm2d(f1), nn.ReLU())
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, 1), nn.BatchNorm2d(f3r), nn.ReLU(),
            nn.Conv2d(f3r, f3, 3, padding=1), nn.BatchNorm2d(f3), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, 1), nn.BatchNorm2d(f5r), nn.ReLU(),
            nn.Conv2d(f5r, f5, 5, padding=2), nn.BatchNorm2d(f5), nn.ReLU())
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_ch, fp, 1), nn.BatchNorm2d(fp), nn.ReLU())

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x),
                          self.branch3(x), self.branch4(x)], dim=1)


class InceptionStyle(nn.Module):
    """
    Inception-style CNN: parallel multi-scale convolutions (1x1, 3x3, 5x5)
    captures LSD lesions at multiple granularities simultaneously.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 192, 3, padding=1),         nn.BatchNorm2d(192), nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        # InceptionModule(in, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5, pool_proj)
        self.inc1 = InceptionModule(192, 64, 96, 128, 16, 32, 32)   # out=256
        self.inc2 = InceptionModule(256, 128, 128, 192, 32, 96, 64) # out=480
        self.pool = nn.MaxPool2d(3, 2, 1)
        self.inc3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)  # out=512
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.pool(x)
        x = self.inc3(x)
        return self.head(x)


# ─────────────────────────────────────────────
# 7. Lightweight CNN  (Depthwise Separable)
# ─────────────────────────────────────────────
def _dw_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                  padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU6(),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU6(),
    )


class LightweightCNN(nn.Module):
    """
    Depthwise separable CNN — minimal parameters, fast inference.
    Suitable for edge deployment on farm/veterinary devices.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(),
            _dw_block(32,  64),
            _dw_block(64,  128, stride=2),
            _dw_block(128, 128),
            _dw_block(128, 256, stride=2),
            _dw_block(256, 256),
            _dw_block(256, 512, stride=2),
            *[_dw_block(512, 512) for _ in range(5)],
            _dw_block(512, 1024, stride=2),
            _dw_block(1024, 1024),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────
def get_model_registry(num_classes):
    return {
        "LeNet-5"          : LeNet5(num_classes),
        "AlexNet-Style"    : AlexNetStyle(num_classes),
        "VGG-3Block"       : VGGStyle(num_classes, num_blocks=3),
        "VGG-4Block"       : VGGStyle(num_classes, num_blocks=4),
        "VGG-5Block"       : VGGStyle(num_classes, num_blocks=5),
        "ResNet-Style"     : ResNetStyle(num_classes),
        "DenseNet-Style"   : DenseNetStyle(num_classes),
        "Inception-Style"  : InceptionStyle(num_classes),
        "Lightweight-CNN"  : LightweightCNN(num_classes),
    }


# ─────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, loaders, sizes, model_name):
    device    = CONFIG["device"]
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG["learning_rate"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"])

    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {"train_acc": [], "val_acc": [],
                 "train_loss": [], "val_loss": []}

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
                        loss.backward(); optimizer.step()
                _, preds = torch.max(outputs, 1)
                running_loss    += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels).item()

            e_loss = running_loss    / sizes[phase]
            e_acc  = running_correct / sizes[phase]
            history[f"{phase}_loss"].append(e_loss)
            history[f"{phase}_acc"].append(e_acc)

            if phase == "val" and e_acc > best_acc:
                best_acc = e_acc
                best_wts = copy.deepcopy(model.state_dict())

        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1:02d}/{CONFIG['num_epochs']}  "
                  f"Val Acc: {history['val_acc'][-1]*100:.2f}%")

    model.load_state_dict(best_wts)
    return model, history, best_acc


def evaluate_model(model, loader):
    device = CONFIG["device"]
    model.eval().to(device)
    all_preds, all_labels = [], []
    total_time = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            t0 = time.time()
            outputs = model(inputs)
            total_time += time.time() - t0
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    avg_inf_ms = (total_time / len(all_preds)) * 1000   # ms per image
    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "inf_ms": avg_inf_ms,
            "preds": all_preds, "labels": all_labels}


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CONFIG["class_names"],
                yticklabels=CONFIG["class_names"], ax=ax)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    fname = os.path.join(CONFIG["results_dir"],
                         title.replace(" ", "_") + "_cm.png")
    plt.savefig(fname, dpi=130); plt.close()


def plot_training_curves(histories):
    """Grid of training curves for all architectures."""
    models  = list(histories.keys())
    n       = len(models)
    cols    = 3
    rows    = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
    axes    = axes.flatten()

    for idx, (name, hist) in enumerate(histories.items()):
        ax = axes[idx]
        ax.plot(hist["train_acc"], "--", color="#e74c3c", alpha=0.7, label="Train")
        ax.plot(hist["val_acc"],   "-",  color="#2ecc71",            label="Val")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Training Curves — All CNN Architectures (LSD Detection)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["results_dir"], "all_training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Training curves → results/all_training_curves.png")


def plot_radar_chart(results):
    """Radar/spider chart comparing architectures on 4 metrics."""
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    N       = len(labels)
    angles  = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw=dict(polar=True))
    cmap    = plt.cm.get_cmap("tab10", len(results))

    for idx, (name, res) in enumerate(results.items()):
        vals = [res["accuracy"], res["precision"],
                res["recall"],   res["f1"]]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=1.8,
                color=cmap(idx), label=name)
        ax.fill(angles, vals, alpha=0.07, color=cmap(idx))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("CNN Architecture Radar Comparison — LSD Detection",
                 fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["results_dir"], "radar_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Radar chart → results/radar_comparison.png")


def plot_metrics_bar(results):
    """Grouped bar chart: Accuracy / F1 / Params / Inference time."""
    names = list(results.keys())
    accs  = [results[n]["accuracy"] * 100  for n in names]
    f1s   = [results[n]["f1"] * 100        for n in names]
    params= [results[n]["params"] / 1e6    for n in names]   # millions
    infs  = [results[n]["inf_ms"]          for n in names]

    x     = np.arange(len(names))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Extensive CNN Investigation — LSD Diagnosis in Dairy Cows\n"
                 "D. K. Saha [7]",
                 fontsize=14, fontweight="bold")

    datasets_bar = [
        (axes[0,0], accs,   "#3498db", "Validation Accuracy (%)",    "Accuracy"),
        (axes[0,1], f1s,    "#2ecc71", "F1-Score (%)",               "F1-Score"),
        (axes[1,0], params, "#e74c3c", "Parameters (Millions)",      "Model Size"),
        (axes[1,1], infs,   "#f39c12", "Avg Inference Time (ms/img)","Speed"),
    ]

    for ax, vals, color, ylabel, title in datasets_bar:
        bars = ax.bar(x, vals, color=color, edgecolor="white",
                      linewidth=0.7, alpha=0.88)
        for bar_, v in zip(bars, vals):
            ax.text(bar_.get_x() + bar_.get_width()/2,
                    bar_.get_height() + max(vals)*0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["results_dir"], "metrics_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Metrics bar chart → results/metrics_comparison.png")


def plot_accuracy_vs_params(results):
    """Scatter: Accuracy vs Model Size — efficiency frontier."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap    = plt.cm.get_cmap("tab10", len(results))

    for idx, (name, res) in enumerate(results.items()):
        ax.scatter(res["params"]/1e6, res["accuracy"]*100,
                   s=120, color=cmap(idx), zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(name,
                    (res["params"]/1e6, res["accuracy"]*100),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color=cmap(idx))

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Model Complexity\n"
                 "LSD Detection — CNN Architecture Investigation",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["results_dir"], "accuracy_vs_params.png"),
                dpi=150)
    plt.close()
    print("  Scatter plot → results/accuracy_vs_params.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "█"*60)
    print("  Extensive CNN Investigation for LSD Diagnosis")
    print("  D. K. Saha — Heliyon 2024  [Reference 7]")
    print("█"*60)

    print("\n[1] Loading Data...")
    loaders, sizes = get_dataloaders()

    registry  = get_model_registry(CONFIG["num_classes"])
    results   = {}
    histories = {}

    print(f"\n[2] Training {len(registry)} CNN Architectures...")

    for model_name, model in registry.items():
        n_params = count_parameters(model)
        print(f"\n{'─'*55}")
        print(f"  Architecture : {model_name}")
        print(f"  Parameters   : {n_params:,}")
        print(f"{'─'*55}")

        trained_model, history, best_acc = train_model(
            model, loaders, sizes, model_name)

        eval_res = evaluate_model(trained_model, loaders["val"])
        eval_res["params"] = n_params

        print(f"\n  Classification Report — {model_name}")
        print(classification_report(eval_res["labels"], eval_res["preds"],
                                    target_names=CONFIG["class_names"]))

        plot_confusion_matrix(eval_res["labels"], eval_res["preds"], model_name)

        results[model_name]   = eval_res
        histories[model_name] = history

        # Save weights
        torch.save(trained_model.state_dict(),
                   os.path.join(CONFIG["save_dir"],
                                f"{model_name.replace(' ','_')}.pth"))

    # ── Plots ────────────────────────────────────
    print("\n[3] Generating Visualisations...")
    plot_training_curves(histories)
    plot_radar_chart(results)
    plot_metrics_bar(results)
    plot_accuracy_vs_params(results)

    # ── Summary Table ────────────────────────────
    print("\n" + "="*80)
    print(f"  {'Architecture':<20} {'Acc%':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'Params(M)':>10} {'ms/img':>8}")
    print("="*80)
    for name, res in sorted(results.items(),
                             key=lambda x: -x[1]["accuracy"]):
        print(f"  {name:<20} "
              f"{res['accuracy']*100:>6.2f}%  "
              f"{res['precision']:>6.4f}  "
              f"{res['recall']:>6.4f}  "
              f"{res['f1']:>6.4f}  "
              f"{res['params']/1e6:>9.2f}M  "
              f"{res['inf_ms']:>7.3f}ms")
    print("="*80)
    best = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\n  Best Architecture : {best}")
    print(f"  Best Val Accuracy : {results[best]['accuracy']*100:.2f}%")
    print(f"  Parameters        : {results[best]['params']:,}")
    print("="*80)


if __name__ == "__main__":
    main()
