import jetson.inference
import jetson.utils
import argparse
import sys
import json

# Parse the command line arguments
parser = argparse.ArgumentParser(
    description="Locate objects in a live camera stream using an object detection DNN.",
    formatter_class=argparse.RawTextHelpFormatter,
    epilog=jetson.inference.detectNet.Usage() +
           jetson.utils.videoSource.Usage() +
           jetson.utils.videoOutput.Usage()
)

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("json_output", type=str, help="Path to save JSON file with detection results")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="Pre-trained model to load")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="Detection overlay flags")
parser.add_argument("--threshold", type=float, default=0.5, help="Minimum detection threshold to use")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
    opt = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# Create video sources and outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv + is_headless)

def detections_to_json(detections):
    """Convert a list of detections to JSON format."""
    detection_list = []
    for detection in detections:
        detection_list.append({
            'ClassID': detection.ClassID,
            'Confidence': detection.Confidence,
            'Left': detection.Left,
            'Top': detection.Top,
            'Right': detection.Right,
            'Bottom': detection.Bottom,
            'Width': detection.Width,
            'Height': detection.Height,
            'Area': detection.Area,
            'Center': (detection.Center[0], detection.Center[1])
        })
    return json.dumps({'detections': detection_list}, indent=4)

# Process frames until the user exits
while True:
    # Capture the next image
    img = input.Capture()

    # Detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=opt.overlay)

    # Print the detections
    print(f"Detected {len(detections)} objects in the image")

    for detection in detections:
        print(detection)

    # Render the image
    output.Render(img)

    # Update the title bar
    output.SetStatus(f"{opt.network} | Network {net.GetNetworkFPS():.0f} FPS")

    # Print out performance info
    net.PrintProfilerTimes()

    # Exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        # Save detections to the specified JSON file
        with open(opt.json_output, 'w') as json_file:
            json_file.write(detections_to_json(detections))
        break
