import subprocess
import json
from PIL import Image
import numpy as np

# Paths
depth_image_path = 'images/test/depth_net_answer.jpg'
bounding_boxes_json_path = 'images/test/detect_net_answer.json'

# Run DepthNet to generate the depth image
depthnet_result = subprocess.run(
    ['./depthnet.py', 'images/cat_2.jpg', depth_image_path])

# Check the results of the first command
if depthnet_result.returncode == 0:
    print("depthnet.py executed successfully.")
else:
    print("depthnet.py error:\n", depthnet_result.stderr)

# Run DetectNet to generate bounding boxes JSON
detectnet_result = subprocess.run(
    ['./detectnet.py', 'images/cat_2.jpg', bounding_boxes_json_path])

# Check the results of the second command
if detectnet_result.returncode == 0:
    print("detectnet.py executed successfully.")
else:
    print("detectnet.py error:\n", detectnet_result.stderr)

# Convert the depth image to a NumPy array
depth_image = Image.open(depth_image_path).convert('L')  # Convert to grayscale
depth_array = np.array(depth_image)

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
    
    # Crop the region of interest from the depth array
    roi = depth_array[top:bottom, left:right]
    
    # Calculate the mean depth value in this region
    mean_depth = np.mean(roi)
    
    # Normalize the mean depth value to a range of [0, 1]
    normalized_depth = mean_depth / 255.0
    
    # Add the mean depth to the detection dictionary
    detection['MeanDepth'] = normalized_depth

# Save the updated detections back to the JSON file
with open(bounding_boxes_json_path, 'w') as f:
    json.dump(bounding_boxes_data, f, indent=4)

print("Updated JSON file with mean depth information.")
