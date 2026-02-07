# Face Mask Detection with Faster R-CNN

Face mask detection system using Faster R-CNN to classify proper mask wearing, no mask, or improper mask wearing.

![Image](https://github.com/user-attachments/assets/65e52fc7-c6bc-47d8-a4ed-4f4edcbf76a1)

## Results

- mAP@0.5: 73.0%
<img width="527" height="79" alt="Image" src="https://github.com/user-attachments/assets/e62796df-12c1-4a19-aa22-6e0efbada90d" />

- Dataset: 853 images from [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
- Training: 26 epochs, validation loss: 0.3081
<img width="592" height="186" alt="Image" src="https://github.com/user-attachments/assets/b4844e84-2169-4144-bdaa-0c69368f0079" />

## Installation
```bash
pip install -r requirements.txt
```

## Usage

Train the model:
```bash
python train.py
```

Run inference (edit `IMAGE_PATH` in test.py):
```bash
python test.py
```

## Model Details

- Faster R-CNN ResNet50-FPN pretrained on COCO
- 3 classes: with_mask, without_mask, mask_weared_incorrect
- Adam optimizer, learning rate 5e-4

## Dataset Structure
```
dataset/
├── images/
└── annotations/  (Pascal VOC XML format)
```

## Files

- `train.py` - Training with validation and mAP evaluation
- `test.py` - Inference with bounding box visualization
- `requirements.txt` - Dependencies
