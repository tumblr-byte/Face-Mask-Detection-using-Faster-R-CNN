import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
import os

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

                print(f"  ✓ Saved: {save_path}")
                sample_count += 1

visualize_test_samples(test_loader, model, device, num_samples=6)
