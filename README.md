# Object Detection Using YOLOv3

This project implements real-time object detection using the **YOLOv3** model. The solution efficiently detects and labels multiple objects in an image with precise bounding boxes.

---

## How Does the Code Work?
The code takes an image and identifies different objects in it using the **YOLOv3** model. It draws colored boxes around these objects and labels them. For example, if you provide a picture of a fruit basket, it can detect apples, oranges, or bananas.

### Key Steps in the Code:
1. **Read the Image:** The image is loaded for processing.
2. **Prepare the Image:** The image is resized and converted into a special format for the YOLO model to understand.
3. **Object Detection:** The model predicts which objects are present and their positions in the image.
4. **Drawing Boxes:** Colored boxes with labels are drawn around detected objects.

---

## What Are Weights?
**Weights** are like the "brain" of the model. They are pre-trained values that help the model recognize objects accurately without you having to train it from scratch.

---

## What is COCO?
**COCO** (Common Objects in Context) is a large collection of images with labeled objects. It's like a huge photo album where each picture has tags like "dog," "cat," "car," etc. The YOLO model uses this data to learn what different objects look like.

---

## Why Are We Using COCO?
COCO helps YOLO recognize common objects easily. Since COCO has thousands of labeled pictures, it trains YOLO to identify things like chairs, phones, or fruits more accurately. By using COCO, our model doesn't need to be trained from scratch, saving time and improving results.

---

## Prerequisites

### **1. Install Dependencies**
```bash
pip install opencv-python numpy argparse
```

### **2. Download YOLOv3 Model Files**
Download the following files and place them in the `coco_weights/` folder:
- **[coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)**
- **[yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)**
- **[yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)**

### **3. Project Structure**
```
object_detection_project/
├── coco_weights/
│   ├── coco.names
│   ├── yolov3.cfg
│   ├── yolov3.weights
├── images/
│   ├── fruits.jpg
├── main.py
└── README.md
```

---

## Usage

### **1. Run the Object Detection Script**
```bash
python3 main.py --image images/fruits.jpg
```

### **2. Command Line Arguments**
- `--image`: Path to the input image (Required)
- `--confidence`: Minimum probability to filter weak detections (Default: 0.5)
- `--threshold`: Non-maxima suppression threshold (Default: 0.3)

**Example Command:**
```bash
python3 main.py --image images/fruits.jpg --confidence 0.6 --threshold 0.4
```

---

## Expected Output
- The detected objects will be highlighted with bounding boxes.
- Each detected object will be labeled with its name and confidence score.

---

## Troubleshooting

1. **Error:** `NameError: name 'ln' is not defined`
   - **Solution:** Ensure this line exists in `main.py`:  
     ```python
     ln = net.getLayerNames()
     ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
     ```

2. **Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'coco_weights/coco.names'`
   - **Solution:** Verify that the `coco.names` file is located in the `coco_weights` folder.

---

## Contact
For questions or support, please reach out to [Your Name] at [Your Email].

