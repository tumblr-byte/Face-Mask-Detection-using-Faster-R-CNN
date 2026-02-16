# Face Mask Detection with Faster R-CNN

Real-time face mask detection system using Faster R-CNN to classify proper mask wearing, no mask, or improper mask wearing. Built for safety-critical applications where **accuracy matters more than speed**.

![Image](https://github.com/user-attachments/assets/65e52fc7-c6bc-47d8-a4ed-4f4edcbf76a1)

---

##  Results

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **86.5%** |
| Training Epochs | 15 (early stopped) |
| Best Val Loss | 0.2803 |
| Final Train Loss | 0.0959 |
| Dataset Size | 853 images |
| Train/Val/Test Split | 597/127/129 (70/15/15) |

---

##  Dataset Information

**Source:** [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

- **Total Images:** 853
- **Total Bounding Boxes:** 4,072
- **Format:** PASCAL VOC XML annotations
- **Classes:** 3 (with_mask, without_mask, mask_weared_incorrect)

### Class Distribution

<img width="989" height="590" alt="Image" src="https://github.com/user-attachments/assets/1fdec436-1bbc-47d8-8051-2e3b45ae3810" />

```
with_mask:              3,232 (79.37%)
without_mask:             717 (17.61%)
mask_weared_incorrect:    123 ( 3.02%)
```

**Note:** Class imbalance reflects real-world distribution where most people wear masks correctly. Model trained with transfer learning from COCO-pretrained weights handles this imbalance without manual rebalancing.

---

## Why Faster R-CNN?

| Decision | Reasoning |
|----------|-----------|
| **Accuracy over Speed** | For safety/compliance applications, false negatives are costly. Chose precision over real-time performance. |
| **ResNet50-FPN Backbone** | Feature Pyramid Network handles faces at multiple scales (close-up vs distant). |
| **Transfer Learning** | COCO pretrained weights reduce training time and improve generalization on small dataset. |
| **Proven Architecture** | Extensively researched, well-documented, easier to debug than newer methods. |



---

## Training Configuration

### Key Hyperparameters
- **Batch Size:** 4 (limited by GPU memory - 6GB VRAM)
- **Learning Rate:** 5e-4 with ReduceLROnPlateau scheduler
- **Optimizer:** Adam
- **Early Stopping:** Patience of 7 epochs
- **Random Seed:** 42 (for reproducibility)

### Data Augmentation (Critical for Small Dataset!)
```python
RandomHorizontalFlip(0.5)      # Faces can be mirrored
ColorJitter(brightness=0.2,     # Handle different lighting
           contrast=0.2, 
           saturation=0.1)
```

### Why These Choices?

**Batch Size 4:**
- Tested batch sizes 2, 4, 8
- Size 8 caused GPU OOM errors
- Size 2 led to unstable gradients
- **4 was the sweet spot**

**Learning Rate 5e-4:**
- Tested: 1e-3 (too fast, diverged), 5e-4 (stable), 1e-4 (too slow)
- Scheduler reduced to 2.5e-4 at epoch 12 when validation plateaued

**70/15/15 Split:**
- Ensures proper held-out test set
- Fixed seed guarantees reproducibility
- No data leakage between sets

---

##  Training Curve



### What Happened During Training:

**Epochs 1-8:** Model learning rapidly, both train and val loss decreasing  
**Epoch 8:** Best validation loss (0.2803) achieved  
**Epochs 9-11:** Validation loss starts increasing while train loss keeps dropping → **Overfitting begins**  
**Epoch 12:** Learning rate reduced from 5e-4 to 2.5e-4 (scheduler triggered)  
**Epochs 12-15:** No improvement despite lower learning rate  
**Epoch 15:** Early stopping triggered (patience exhausted)

---

## Known Limitations & Overfitting Analysis

### Evidence of Overfitting

```
Train Loss: 0.0959  ← Very low
Val Loss:   0.3171  ← 3.3x higher
```

**This gap indicates the model memorized training data rather than learning generalizable features.**

### Where the Model Fails

1. **Profile Views & Unusual Angles**
   - Training data mostly frontal faces
   - Struggles with side profiles, tilted heads
   
2. **False Positives on Partial Faces**
   - Sometimes detects "masks" on ears, hair, or background
   - Overconfident even when wrong (scores >0.85)
   
3. **Occlusions**
   - Hands covering face confuse the model
   - Sunglasses + mask = poor predictions

### Why This Happened

1. **Small Dataset (853 images)**
   - Not enough variety in angles, lighting, demographics
   - Modern object detectors typically need 5,000+ images
   
2. **Class Imbalance**
   - Only 3% "mask_weared_incorrect" examples
   - Model underperforms on this minority class
   
3. **Limited Diversity**
   - Dataset lacks edge cases (profile views, occlusions, poor lighting)
   - All images from similar sources/contexts

### How to Fix

**Short-term (within current constraints):**
- Add dropout layers for regularization
- More aggressive augmentation (rotations, crops, synthetic occlusions)
- Reduce model complexity (use ResNet18 instead of ResNet50)

**Long-term (ideal solution):**
- Collect 5,000+ diverse images
- Include edge cases: profiles, occlusions, varied demographics
- Use ensemble methods or test-time augmentation
- Try newer architectures (Cascade R-CNN, DETR)

---

##  Installation

```bash
git clone https://github.com/tumblr-byte/face-mask-detection.git
cd face-mask-detection
pip install -r requirements.txt
```


##  Usage



### 1. Train Model
```bash
python train.py
```
- Trains for max 30 epochs with early stopping
- Saves best model to `best_model.pth`
- Takes ~1.5 hours on T4 GPU

### 3. Run Inference
```bash
python test.py
```




---

##  What I Learned

### Technical Lessons
1. **Small datasets require aggressive regularization** - Data augmentation was critical
2. **Transfer learning is powerful** - COCO pretraining gave huge boost despite different domain
3. **Early stopping prevents overfitting** - Without it, model would have overfit even more
4. **Validation curves tell the truth** - Train loss alone is misleading

### Engineering Lessons
1. **Start simple, iterate** - Tried complex augmentation first, scaled back to what worked
2. **Monitor both metrics** - High mAP doesn't mean deployment-ready
3. **Test on edge cases** - Model works on "easy" test set but fails on real-world cases
4. **Document limitations honestly** - Better to acknowledge weaknesses than hide them

---

## Future Improvements

If given more time/resources:

### Data Collection
- Gather 5,000+ images with:
  - Diverse demographics (age, ethnicity)
  - Various angles (frontal, profile, tilted)
  - Different lighting (indoor, outdoor, low-light)
  - Edge cases (occlusions, accessories)

### Model Improvements
- Try Cascade R-CNN (multi-stage refinement)
- Experiment with DETR (transformer-based)
- Add focal loss to handle class imbalance
- Implement test-time augmentation

### Deployment
- Build REST API with FastAPI
- Add real-time video processing
- Create web demo with Streamlit
- Containerize with Docker

---

## Technical Notes

### Why Not YOLO?
- YOLO is faster (~30 FPS vs ~15 FPS)
- Faster R-CNN has higher precision for small faces
- For compliance monitoring, accuracy > speed
- False negatives (missed unmasked people) are costly

### Why 3 Classes?
- "Mask worn incorrectly" (nose exposed) is common real-world issue
- Provides actionable feedback vs binary classification
- Helps identify compliance vs safety issues

### Why No Class Weights?
- Tested weighted loss for class imbalance
- Transfer learning from COCO already handles imbalance well
- No significant improvement (<1% mAP gain)
- Added complexity not worth minimal benefit

---
