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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms


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
            else:  # mask_weared_incorrect
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


# CONFIGURATION - CHANGE THESE
DATASET_PATH = path  # Your dataset root folder
BATCH_SIZE = 4  # Kept at 4 to avoid GPU OOM (out of memory)
EPOCHS = 30
LEARNING_RATE = 5e-4
PATIENCE = 7  # Early stopping patience
SEED = 42  # For reproducibility

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# DATA AUGMENTATION
train_transform = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Vary lighting
    T.RandomHorizontalFlip(0.5),  # Flip faces
    T.ToTensor()
])

val_transform = T.Compose([
    T.ToTensor()  # No augmentation for validation
])


full_dataset = CustomDataset(img_dir=os.path.join(DATASET_PATH, "images"),  ann_dir=os.path.join(DATASET_PATH, "annotations"), transforms=None)

# 70% train, 15% val, 15% test split
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))

# Apply transforms
train_dataset.dataset.transforms = train_transform
val_dataset.dataset.transforms = val_transform
test_dataset.dataset.transforms = val_transform

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))


# MODEL SETUP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Replace classifier head for 4 classes (background + 3 mask classes)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

model.to(device)

# Optimizer and scheduler
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)


# Reduce learning rate when validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# TRAINING FUNCTION
def train_model(model, optimizer, scheduler, train_loader, val_loader, device, epochs=30, patience=7, save_path="best_model.pth"):
    best_loss = float("inf")
    counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # TRAINING 
        model.train()
        train_loss = 0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # VALIDATION
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Need to set model to train mode to get loss
                model.train()
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
                model.eval()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']


        print(f"Epoch {epoch+1}/{epochs} , Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        # EARLY STOPPING
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Model saved (best val loss: {best_loss:.4f})")
        else:
            counter += 1
            print(f"No improvement")

        if counter >= patience:
            print(f"Early stopping triggered")
            break

    return history


history = train_model(model, optimizer, scheduler, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE, save_path="best_model.pth")

# EVALUATION FUNCTION
def evaluate_map(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Computing mAP"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            metric.update(outputs, targets)

    results = metric.compute()
    print(f"mAP@0.5 (Overall): {results['map_50'].item():.4f}")

    return results['map_50'].item()
    

model.load_state_dict(torch.load("best_model.pth"))
test_map50 = evaluate_map(model, test_loader, device)


# Visualize_test_samples
def visualize_test_samples(test_loader, model, device, num_samples=6):
    os.makedirs("comparison_results", exist_ok=True)
    model.eval()
    class_names = {1: 'with_mask', 2: 'without_mask', 3: 'mask_weared_incorrect'}
    colors = {1: 'green', 2: 'red', 3: 'orange'}

    sample_count = 0

    with torch.no_grad():
        for images, targets in test_loader:
            if sample_count >= num_samples:
                break

            for img, target in zip(images, targets):
                if sample_count >= num_samples:
                    break

                img_tensor = img.to(device)

                # Get predictions
                pred = model([img_tensor])[0]
                pred_boxes = pred['boxes']
                pred_labels = pred['labels']
                pred_scores = pred['scores']

                # Filter and NMS
                mask = pred_scores >= 0.6
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]

                if len(pred_boxes) > 0:
                    keep = nms(pred_boxes, pred_scores, 0.5)
                    pred_boxes = pred_boxes[keep].cpu()
                    pred_labels = pred_labels[keep].cpu()
                    pred_scores = pred_scores[keep].cpu()
                else:
                    pred_boxes = pred_boxes.cpu()
                    pred_labels = pred_labels.cpu()
                    pred_scores = pred_scores.cpu()

                # Ground truth
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()

                # Create visualization
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                # Ground Truth
                ax1.imshow(img_np)
                ax1.set_title('Ground Truth', fontsize=20, fontweight='bold', color='blue')

                for box, label in zip(gt_boxes, gt_labels):
                    x_min, y_min, x_max, y_max = box
                    color = colors.get(label.item(), 'blue')
                    label_text = class_names[label.item()]

                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=3, edgecolor=color, facecolor='none'
                    )
                    ax1.add_patch(rect)
                    ax1.text(x_min, y_min - 10, label_text, color='white',
                            backgroundcolor=color, fontsize=12, fontweight='bold')

                ax1.axis('off')

                # Predictions
                ax2.imshow(img_np)
                ax2.set_title('Model Predictions', fontsize=20, fontweight='bold', color='darkgreen')

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    x_min, y_min, x_max, y_max = box
                    color = colors.get(label.item(), 'blue')
                    label_text = f"{class_names[label.item()]}: {score:.2f}"

                    rect = patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=3, edgecolor=color, facecolor='none'
                    )
                    ax2.add_patch(rect)
                    ax2.text(x_min, y_min - 10, label_text, color='white',
                            backgroundcolor=color, fontsize=12, fontweight='bold')

                ax2.axis('off')

                plt.tight_layout()
                save_path = f"comparison_results/comparison_{sample_count+1}.jpg"
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()

                print(f"Saved: {save_path}")
                sample_count += 1

visualize_test_samples(test_loader, model, device, num_samples=6)

