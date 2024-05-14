pip install timm

import torchvision.transforms as T
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from google.colab import files
from google.colab.patches import cv2_imshow
# Load MiDaS model
model_type = "DPT_Large"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load MiDaS transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform_midas = midas_transforms.dpt_transform
else:
    transform_midas = midas_transforms.small_transform

# Function to upload a file
def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"User uploaded file '{filename}' with length {len(uploaded[filename])} bytes")
    return next(iter(uploaded))
image_path = upload_image()

# MiDaS depth estimation
img_midas = cv2.imread(image_path)
img_midas = cv2.cvtColor(img_midas, cv2.COLOR_BGR2RGB)

# Apply MiDaS transforms and predict depth
input_batch = transform_midas(img_midas).to(device)
with torch.no_grad():
    prediction_midas = midas(input_batch)
    prediction_midas = torch.nn.functional.interpolate(
        prediction_midas.unsqueeze(1),
        size=img_midas.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction_midas.cpu().numpy()

# Show depth map
plt.imshow(depth_map, cmap='inferno')
plt.axis('off')
plt.show()

# Proceed with object detection on the original image
image = Image.open(image_path).convert("RGB")
transform = T.Compose([T.ToTensor()])
img = transform(image).to(device)

# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
model.to(device)
model.eval()

with torch.no_grad():
    prediction = model([img])

# Helper function for non-maximum suppression
def nms(boxes, scores, iou_threshold):
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.2, nms_threshold=iou_threshold)

    if indices is None or len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    return indices.tolist()

for i, (box, score, label) in enumerate(zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels'])):
    print(f"Object {i+1}: Score: {score.item():.2f}")

# Process and visualize object detection results
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

boxes = [box.cpu().numpy().astype(int) for box in prediction[0]['boxes']]
scores = [score.cpu().numpy() for score in prediction[0]['scores']]
masks = [mask[0].cpu().numpy() for mask in prediction[0]['masks']]

selected_indices = nms(boxes, scores, iou_threshold=0.2)
def annotate_image(image, text, position, font_scale=0.5, font_thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

for idx in selected_indices:
    box = boxes[idx]
    score = scores[idx]
    mask = masks[idx] > 0.2  # Binarize the mask

    # Draw bounding box
    cv2.rectangle(image_cv, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_cv, contours, -1, (0, 255, 0), 2)

    # Check if contours were found and annotate the image
    if contours:
        area = cv2.contourArea(contours[0])  # Compute the area of the first contour
        annotate_image(image_cv, f"Area: {area:.2f}px", (box[0], box[1] - 10))  # Annotate image with area

# Display the processed image with bounding boxes, mask edges, and depth map
cv2_imshow(image_cv)

