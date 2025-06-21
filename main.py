import os
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import torch.utils.data
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

path = "path to your dataset"
# Custom Dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, annotation_dir, transforms=None):
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms

        # Get list of images and annotations
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        print(f"Found {len(self.images)} images")

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        annotation_file = os.path.join(
            self.annotation_dir,
            self.images[idx].replace('.jpg', '.xml').replace('.jpeg', '.xml').replace('.png', '.xml')
        )

        # Parse XML annotation
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        boxes = []
        labels = []

        # Extract bounding boxes and labels
        for obj in root.findall('object'):
            # Get label
            label = obj.find('name').text
            # Convert label to number
            if label == 'with_mask':
                label_id = 1
            elif label == 'without_mask':
                label_id = 2
            else:
                label_id = 3  # incorrect mask

            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label_id)

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.images)

# Define transformations
transform = T.Compose([
    T.ToTensor(),
])
print("done")



dataset = CustomDataset(
    img_dir= os.path.join(path ,"images") ,
    annotation_dir= os.path.join(path , "annotations"),

    transforms=transform
)


train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=2,
    collate_fn=lambda x: tuple(zip(*x))
)



# Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 4 
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0005)


# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()  

    print(f"[Epoch {epoch+1}] Train Loss: {train_loss / len(train_loader):.4f}")
