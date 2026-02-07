import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm



class CustomDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        ann_path = os.path.join(
            self.ann_dir,
            self.images[idx].rsplit(".", 1)[0] + ".xml"
        )

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes, labels = [], []

        for obj in root.findall("object"):
            name = obj.find("name").text
            if name == "with_mask":
                label = 1
            elif name == "without_mask":
                label = 2
            else:
                label = 3

            b = obj.find("bndbox")
            boxes.append([
                float(b.find("xmin").text),
                float(b.find("ymin").text),
                float(b.find("xmax").text),
                float(b.find("ymax").text),
            ])
            labels.append(label)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target



path = path  # dataset root

transform = T.Compose([T.ToTensor()])

dataset = CustomDataset(
    img_dir=os.path.join(path, "images"),
    ann_dir=os.path.join(path, "annotations"),
    transforms=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset= random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset,batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
model.to(device)

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-4)

def train_model(model, optimizer, train_loader, val_loader, device, epochs=30, patience=7, save_path="best.pth"):
    best_loss = float("inf")
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Train {epoch+1}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                model.train()  
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
                model.eval()

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print("Model saved")
        else:
            counter += 1
            print("No improvement")

        if counter >= patience:
            print("Early stopping")
            break



train_model(model,optimizer, train_loader, val_loader, device, epochs=30, patience=7, save_path="best.pth")

def evaluate_map(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="mAP"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)

    return metric.compute()["map_50"].item()


model.load_state_dict(torch.load("best.pth"))
map50 = evaluate_map(model, val_loader, device)
print(f"mAP@0.5: {map50:.4f}")
