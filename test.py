import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# CONFIGURATION - CHANGE THESE
IMAGE_PATH = "test_image.jpg"  # UPDATE THIS PATH
MODEL_PATH = "best_model.pth"
THRESHOLD = 0.6
NMS_THRESHOLD = 0.5


# LOAD MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# RUN INFERENCE
print(f"Running inference on: {IMAGE_PATH}")
img = Image.open(IMAGE_PATH).convert("RGB")
transform = T.Compose([T.ToTensor()])
img_tensor = transform(img).to(device)

with torch.no_grad():
    pred = model([img_tensor])[0]

boxes = pred['boxes']
labels = pred['labels']
scores = pred['scores']

# Filter by confidence
mask = scores >= THRESHOLD
boxes = boxes[mask]
labels = labels[mask]
scores = scores[mask]

# Apply NMS to remove double detections
if len(boxes) > 0:
    keep = nms(boxes, scores, NMS_THRESHOLD)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

boxes = boxes.cpu()
labels = labels.cpu()
scores = scores.cpu()

print(f"Detected {len(boxes)} faces")


# VISUALIZATION
class_names = {1: 'with_mask', 2: 'without_mask', 3: 'mask_weared_incorrect'}
colors = {1: 'green', 2: 'red', 3: 'orange'}

img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img_np)

for box, label, score in zip(boxes, labels, scores):
    x_min, y_min, x_max, y_max = box
    color = colors.get(label.item(), 'blue')
    label_text = f"{class_names[label.item()]}: {score:.2f}"
    
    # Draw box
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=3, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    
    # Draw label
    ax.text(
        x_min, y_min - 10, label_text,
        color='white', backgroundcolor=color,
        fontsize=14, fontweight='bold'
    )
    
    print(f"  - {label_text} at ({x_min:.0f}, {y_min:.0f})")

plt.axis('off')
plt.tight_layout()
plt.savefig('output.jpg', bbox_inches='tight', dpi=150)
print(f"Saved result to: output.jpg")
plt.show()
