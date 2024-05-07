import gradio as gr
import subprocess
import json
from PIL import Image
import numpy as np
import cv2


# List of class names in order (replace with your own list if different)
class_names = [
    "unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "street sign",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack",
    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def process_image(input_image):
    depth_image_path = 'depth_net_answer.jpg'
    bounding_boxes_json_path = 'detect_net_answer.json'
    output_image_path = 'detect_net_answer_with_depth.jpg'
    
    try:
        # Save the input image locally
        input_image_path = 'input_image.jpg'
        input_image.save(input_image_path)

        # Run DepthNet
        depthnet_result = subprocess.run(
            ['./depthnet.py', input_image_path, depth_image_path],
            capture_output=True, text=True)
        if depthnet_result.returncode != 0:
            return f"depthnet.py error:\n{depthnet_result.stderr}"

        # Run DetectNet
        detectnet_result = subprocess.run(
            ['./detectnet.py', input_image_path, bounding_boxes_json_path],
            capture_output=True, text=True)
        if detectnet_result.returncode != 0:
            return f"detectnet.py error:\n{detectnet_result.stderr}"

        # Convert depth image to a NumPy array
        depth_image = Image.open(depth_image_path).convert('L')
        depth_array = np.array(depth_image)

        # Load the original image using OpenCV
        original_image = cv2.imread(input_image_path)

        # Read bounding box coordinates from the JSON file
        with open(bounding_boxes_json_path, 'r') as f:
            bounding_boxes_data = json.load(f)

        # Draw bounding boxes and depth information
        for detection in bounding_boxes_data['detections']:
            left = int(detection['Left'])
            top = int(detection['Top'])
            right = int(detection['Right'])
            bottom = int(detection['Bottom'])
            class_id = detection['ClassID']

            roi = depth_array[top:bottom, left:right]
            mean_depth = np.mean(roi)
            normalized_depth = mean_depth / 255.0
            detection['MeanDepth'] = normalized_depth

            cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{class_names[class_id]}: {normalized_depth:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_top_left = (right - label_size[0] - 5, bottom - 5)
            label_bottom_right = (right - 5, bottom - label_size[1] - 5)
            cv2.rectangle(original_image, label_top_left, label_bottom_right, (0, 255, 0), cv2.FILLED)
            cv2.putText(original_image, label, (right - label_size[0] - 3, bottom - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Save the final annotated image
        cv2.imwrite(output_image_path, original_image)

        return output_image_path
    except Exception as e:
        return str(e)


#


# Gradio Markdown content
title = "Depth and Detection Analysis"
description = """
This application uses pre-trained models to analyze an input image. It identifies objects and computes their approximate depth using **DepthNet** and **DetectNet**.

### Features:
1. **Object Detection:** Draws bounding boxes and labels around detected objects.
2. **Depth Estimation:** Computes the mean depth of each detected object.

**Instructions:**
1. Upload an image to be analyzed.
2. Click 'Submit'.
3. View and download the annotated image.
"""

# Create Gradio interface
interface = gr.Interface(
    fn=process_image, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Image(type="pil"),
    title=title,
    description=description,
    
)

# Launch Gradio app
interface.launch()
