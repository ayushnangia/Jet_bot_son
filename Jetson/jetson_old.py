import subprocess
import json
from PIL import Image
import numpy as np
import cv2

# Paths
input_image_path = 'images/cat_2.jpg'
output_image_path = 'images/test/detect_net_answer_with_depth.jpg'
depth_image_path = 'images/test/depth_net_answer.jpg'
bounding_boxes_json_path = 'images/test/detect_net_answer.json'

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

# Run DepthNet to generate the depth image
depthnet_result = subprocess.run(
    ['./depthnet.py', input_image_path, depth_image_path])

# Check the results of the first command
if depthnet_result.returncode == 0:
    print("depthnet.py executed successfully.")
else:
    print("depthnet.py error:\n", depthnet_result.stderr)

# Run DetectNet to generate bounding boxes JSON
detectnet_result = subprocess.run(
    ['./detectnet.py', input_image_path, bounding_boxes_json_path])

# Check the results of the second command
if detectnet_result.returncode == 0:
    print("detectnet.py executed successfully.")
else:
    print("detectnet.py error:\n", detectnet_result.stderr)

# Convert the depth image to a NumPy array
depth_image = Image.open(depth_image_path).convert('L')  # Convert to grayscale
depth_array = np.array(depth_image)

# Load the original image using OpenCV
original_image = cv2.imread(input_image_path)

# Read bounding box coordinates from the JSON file
with open(bounding_boxes_json_path, 'r') as f:
    bounding_boxes_data = json.load(f)

# Iterate through each detection and calculate the mean depth
for detection in bounding_boxes_data['detections']:
    # Extract bounding box coordinates
    left = int(detection['Left'])
    top = int(detection['Top'])
    right = int(detection['Right'])
    bottom = int(detection['Bottom'])
    class_id = detection['ClassID']
    
    # Crop the region of interest from the depth array
    roi = depth_array[top:bottom, left:right]
    
    # Calculate the mean depth value in this region
    mean_depth = np.mean(roi)
    
    # Normalize the mean depth value to a range of [0, 1]
    normalized_depth = mean_depth / 255.0
    
    # Add the mean depth to the detection dictionary
    detection['MeanDepth'] = normalized_depth
    
    # Draw the bounding box on the original image using OpenCV
    cv2.rectangle(original_image, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{class_names[class_id]}: {normalized_depth:.2f}"
    
    # Display the label within the bounding box at the bottom-right corner
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_top_left = (right - label_size[0] - 5, bottom - 5)
    label_bottom_right = (right - 5, bottom - label_size[1] - 5)
    cv2.rectangle(original_image, label_top_left, label_bottom_right, (0, 255, 0), cv2.FILLED)
    cv2.putText(original_image, label, (right - label_size[0] - 3, bottom - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Save the final annotated image
cv2.imwrite(output_image_path, original_image)

print("Annotated image saved with bounding boxes, class IDs, and mean depth information.")
