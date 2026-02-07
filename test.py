import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#config
IMAGE_PATH = ""  # Change this
MODEL_PATH = "best.pth"
THRESHOLD = 0.6


#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


img = Image.open(IMAGE_PATH).convert("RGB")
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img).to(device)

with torch.no_grad():
    pred = model([img_tensor])[0]

boxes = pred['boxes'].cpu()
labels = pred['labels'].cpu()
scores = pred['scores'].cpu()
class_names = {1: 'with_mask', 2: 'without_mask', 3: 'mask_weared_incorrect'}
colors = {1: 'green', 2: 'red', 3: 'orange'}


img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img_np)

for box, label, score in zip(boxes, labels, scores):
    if score < THRESHOLD:
        continue
    
    x_min, y_min, x_max, y_max = box
    color = colors.get(label.item(), 'blue')
    label_text = f"{class_names[label.item()]}: {score:.2f}"
    
    # Draw box
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=3, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    

    ax.text(
        x_min, y_min - 10, label_text,
        color='white', backgroundcolor=color,
        fontsize=14, fontweight='bold'
    )

plt.axis('off')
plt.savefig('output.jpg', bbox_inches='tight', dpi=150)
plt.show()
