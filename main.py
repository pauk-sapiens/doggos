# train_production_macos_resnet18.py

import os
import xml.etree.ElementTree as ET
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from PIL import Image
from torchvision.datasets import ImageFolder

# -------------------------
# 1. Dataset with BBox cropping
# -------------------------
class StanfordDogsBBox(ImageFolder):
    def __init__(self, root, ann_dir, transform=None):
        super().__init__(root, transform=transform)
        self.ann_dir = ann_dir

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        breed = os.path.basename(os.path.dirname(img_path))
        xml_name = os.path.splitext(os.path.basename(img_path))[0] + ".xml"
        xml_path = os.path.join(self.ann_dir, breed, xml_name)

        img = Image.open(img_path).convert("RGB")
        try:
            tree = ET.parse(xml_path)
            bbox = tree.find(".//bndbox")
            x1, y1 = int(bbox.find("xmin").text), int(bbox.find("ymin").text)
            x2, y2 = int(bbox.find("xmax").text), int(bbox.find("ymax").text)
            img = img.crop((x1, y1, x2, y2))
        except Exception:
            pass

        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    # -------------------------
    # 2. Configuration
    # -------------------------
    DATA_DIR    = "dataset"
    ANN_DIR     = os.path.join(DATA_DIR, "Annotation")
    TRAIN_DIR   = os.path.join(DATA_DIR, "train")
    TEST_DIR    = os.path.join(DATA_DIR, "test")

    NUM_EPOCHS  = 40           # more epochs for higher accuracy
    BATCH_SIZE  = 64
    LR          = 3e-4
    PATIENCE    = 7
    VAL_RATIO   = 0.1

    # detect device, prefer MPS on macOS, then CUDA, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # DataLoader tuning
    NUM_WORKERS = max(1, os.cpu_count() // 2)
    PIN_MEMORY  = (device.type == "cuda")

    # normalization (ImageNet)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # -------------------------
    # 3. Transforms with strong augmentation
    # -------------------------
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.IMAGENET),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    # -------------------------
    # 4. Datasets & DataLoaders
    # -------------------------
    full_ds = StanfordDogsBBox(TRAIN_DIR, ANN_DIR, transform=train_tf)
    n_val   = int(len(full_ds) * VAL_RATIO)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    val_ds.dataset.transform = val_tf

    test_ds = StanfordDogsBBox(TEST_DIR, ANN_DIR, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=True
    )

    # -------------------------
    # 5. Model, Opt, Scheduler
    # -------------------------
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )
    # Freeze all except layer4 & fc
    for name, param in model.named_parameters():
        if not (name.startswith("layer4") or name.startswith("fc")):
            param.requires_grad = False

    num_classes = len(full_ds.classes)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
        nn.LogSoftmax(dim=1),
    )
    model = model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler    = GradScaler()  # mixed precision

    # -------------------------
    # 6. Early Stopping
    # -------------------------
    class EarlyStopping:
        def __init__(self, patience=7, delta=0.0, path="best_model.pt"):
            self.patience  = patience
            self.delta     = delta
            self.best_loss = np.inf
            self.counter   = 0
            self.path      = path

        def step(self, val_loss, model):
            if val_loss < self.best_loss - self.delta:
                self.best_loss = val_loss
                self.counter   = 0
                torch.save(model.state_dict(), self.path)
            else:
                self.counter += 1
            return self.counter >= self.patience

    early_stopper = EarlyStopping(patience=PATIENCE, delta=0.01, path="best_model.pt")

    # -------------------------
    # 7. Training Loop
    # -------------------------
    for epoch in range(1, NUM_EPOCHS+1):
        # — Train —
        model.train()
        tloss = tcorrect = ttotal = 0
        t0 = timer()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast():
                out  = model(imgs)
                loss = criterion(out, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tloss    += loss.item() * imgs.size(0)
            preds     = out.argmax(1)
            tcorrect += (preds == lbls).sum().item()
            ttotal   += lbls.size(0)

        # — Validate —
        model.eval()
        vloss = vcorrect = vtotal = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                l  = criterion(out, lbls)
                vloss    += l.item() * imgs.size(0)
                preds     = out.argmax(1)
                vcorrect += (preds == lbls).sum().item()
                vtotal   += lbls.size(0)

        tloss /= ttotal; tacc = tcorrect/ttotal*100
        vloss /= vtotal;  vacc = vcorrect/vtotal*100
        scheduler.step()
        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
              f"Train: loss={tloss:.4f}, acc={tacc:.2f}% | "
              f"Val:   loss={vloss:.4f}, acc={vacc:.2f}% | "
              f"Time={(timer()-t0):.1f}s")

        if early_stopper.step(vloss, model):
            print("Early stopping. Loading best model.")
            break

    # load best weights
    model.load_state_dict(torch.load("best_model.pt"))

    # -------------------------
    # 8. Quantize & Export
    # -------------------------
    q_model   = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    scripted  = torch.jit.script(q_model)
    scripted.save("model_prod.pt")

    dummy = torch.randn(1,3,224,224, device=device)
    torch.onnx.export(
        scripted, dummy, "model_prod.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=12, do_constant_folding=True
    )

    print("\n✅ Done. Saved best_model.pt, model_prod.pt & model_prod.onnx")

if __name__ == "__main__":
    main()
