"""
EcoSort AI - training script (run ONCE, separately from the website).
Change the settings in the SETTINGS section below, then run:  python train.py

Dataset layout (one folder per class, folder name = class name):
    dataset/
        class_a/  img1.jpg ...
        ... (6 folders in total)
Optional: if you already have dataset/train and dataset/val, those are used.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ======================= SETTINGS (edit here) =======================
DATASET_PATH = r"C:\Desktop\VS code 44\PyTorch\ECO_SORT_AI\dataset"  # <- your dataset folder
EPOCHS = 10
BATCH_SIZE = 32          # use 16 if you run out of memory
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
MODEL_SAVE_PATH = "model/model.pth"
# ====================================================================

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # must match class_names.py

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def main():
    # ---- Load data ----
    if os.path.isdir(os.path.join(DATASET_PATH, "train")) and os.path.isdir(os.path.join(DATASET_PATH, "val")):
        train_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), train_tf)
        val_ds = datasets.ImageFolder(os.path.join(DATASET_PATH, "val"), val_tf)
        class_names = train_ds.classes
    else:
        # No split provided: make an 80/20 split of the single folder.
        full_train = datasets.ImageFolder(DATASET_PATH, train_tf)
        full_val = datasets.ImageFolder(DATASET_PATH, val_tf)
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(full_train), generator=g).tolist()
        cut = int(0.8 * len(idx))
        train_ds, val_ds = Subset(full_train, idx[:cut]), Subset(full_val, idx[cut:])
        class_names = full_train.classes

    assert len(class_names) == 6, f"Expected 6 class folders, found {len(class_names)}: {class_names}"

    # num_workers=0 avoids multiprocessing problems on Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # ---- Model: pretrained ResNet18 with a new 6-class output layer ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # Training pass
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        # Validation pass
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch}/{EPOCHS}  loss {running / len(train_ds):.4f}  val acc {acc:.3f}")

        if acc >= best_acc:                       # keep the best version
            best_acc = acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nDone. Best validation accuracy: {best_acc:.3f} -> saved to {MODEL_SAVE_PATH}")
    print("\nPaste this into model/class_names.py (order matters!):")
    print("CLASS_NAMES =", class_names)


if __name__ == "__main__":      # required on Windows for DataLoader
    main()
