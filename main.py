# Import Libraries
import numpy as np
import cv2
import argparse
import os

# Argument Parser for CLI Commands
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
parser.add_argument("--confidence", type=float, default=0.5, help="Min confidence")
parser.add_argument("--threshold", type=float, default=0.3, help="NMS threshold")
args = vars(parser.parse_args())

# Load COCO Class Labels
labels_path = "coco-weighs/coco.names"
with open(labels_path, "r") as file:
    LABELS = file.read().strip().split("\n")

# Load YOLO Model's Weights and Config
weights_path = "coco-weighs/yolov3.weights"
config_path = "coco-weighs/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load Input Image
image = cv2.imread(args["image"])
height, width = image.shape[:2]

# YOLO Layer Extraction
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Convert Image to Blob Format for YOLO Processing
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
layer_outputs = net.forward(output_layers)

# Initialize Lists for Detected Objects
boxes, confidences, class_ids = [], [], []
# Loop Over Each Detection
for output in layer_outputs:
    for detection in output:
        scores = detection[5:] # Extract confidence scores
        class_id = np.argmax(scores) # Find the highest confidence class
        confidence = scores[class_id] # Confidence score of the selected class
        # Filter by Confidence Threshold
        if confidence > args["confidence"]:
            box = detection[0:4] * np.array([width, height, width, height])
            centerX, centerY, w, h = box.astype("int") # Extract box details
            x, y = int(centerX - (w / 2)), int(centerY - (h / 2)) # Calculate top-left corner
            
            # Store Detection Information
            boxes.append([x, y, int(w), int(h)])
            confidences.append(float(confidence))
            class_ids.append(class_id)


# Apply Non-Maximum Suppression (NMS) to Refine Bounding Boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])


# Draw Bounding Boxes on Image
if len(idxs) > 0:
    for i in idxs.flatten():
        x, y, w, h = boxes[i]
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show Final Result
cv2.imshow("Detected Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()