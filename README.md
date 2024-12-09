# Floorplan-Segmentation-and-Bedroom-Mapping
This project involves training a Mask R-CNN model to segment elements in 3D floor plans of bedrooms. The model is trained using TensorFlow 2 and can predict masks for objects in unseen images.

## **1. Installation**
To set up the required dependencies, run:
```bash
pip install tensorflow==2.9.1 keras opencv-python matplotlib scikit-image numpy
```
## **2. Dataset Preparation**
Store your dataset in a folder named dataset/.
The dataset should be in COCO-like format with the following structure:

- dataset/
  - train/
    - images/
    - annotations.json
  - val/
    - images/
    - annotations.json

## 3. Training the Model
Open train.ipynb.
Make sure your dataset path is correct.
Run the following code to train the model:
```python
# Import required libraries
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Set up configuration

class TrainConfig(Config):
    NAME = "object"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + Object
    STEPS_PER_EPOCH = 100

config = TrainConfig()
```
# Load the model for training
```python
model = modellib.MaskRCNN(mode="training", config=config, model_dir="./logs")
```
# Path to pre-trained weights
```python
COCO_WEIGHTS_PATH = "./mask_rcnn_coco.h5"
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
```
# Train the model
```python
model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=30, layers="all")
```
## 4. Testing and Evaluating the Model

1. Open test.ipynb.
2. Use the following code to load the trained model for inference:
```python
# Set up inference configuration
class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Path to the trained weights
model_path = "logs/object20241208T1226/mask_rcnn_object_0025.h5"

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir="./logs")
model.load_weights(model_path, by_name=True)
```
3. Evaluate the model:
```python
from mrcnn.utils import compute_ap

def evaluate_model(model, dataset, inference_config, iou_threshold=0.5):
    APs = []
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config, image_id, use_mini_mask=False)
        results = model.detect([image], verbose=0)
        r = results[0]
        AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r["masks"])
        APs.append(AP)
    mean_ap = np.mean(APs)
    return mean_ap

mean_ap = evaluate_model(model, val_dataset, inference_config, iou_threshold=0.5)
print("Mean Average Precision (mAP):", mean_ap)
```
4. Visualize the predictions
```python
from mrcnn import visualize

# Visualize predictions for a single image
image_id = random.choice(val_dataset.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(val_dataset, inference_config, image_id, use_mini_mask=False)
results = model.detect([image], verbose=0)
r = results[0]

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], val_dataset.class_names, r['scores'])
```
# 5. Logs and Weights
Trained weights are stored in the logs/ directory.
The naming convention of weights is mask_rcnn_object_{epoch:04d}.h5.

# 6. Results Visualization
You can save the visualization results with the following code:
```python
import os
from mrcnn import visualize

output_dir = "results/"
os.makedirs(output_dir, exist_ok=True)

for image_id in val_dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(val_dataset, inference_config, image_id, use_mini_mask=False)
    results = model.detect([image], verbose=0)
    r = results[0]
    file_path = os.path.join(output_dir, f"result_{image_id}.jpg")
    visualize.save_instances(image, r['rois'], r['masks'], r['class_ids'], val_dataset.class_names, r['scores'], file_path=file_path)
```
# 7. Key Metrics
Mean Average Precision (mAP): The model achieved a Mean Average Precision (mAP) of [Insert mAP value] at an IoU threshold of 0.5.

