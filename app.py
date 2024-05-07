import gradio as gr
import time
import os
import uuid
from PIL import Image
import numpy as np
import cv2

# Placeholder image path or URL
placeholder_image_path = 'images.png'
try:
    from jetbot import Robot, Camera, bgr8_to_jpeg
    robot = Robot()
    camera = Camera.instance()
except ImportError as e:
    print(f"Import error: {e}\nRunning in simulation mode.")
    robot = None
    camera = None

def move_forward():
    if robot:
        robot.forward(0.3)
        time.sleep(1)
        robot.stop()

def move_backward():
    if robot:
        robot.backward(0.3)
        time.sleep(1)
        robot.stop()

def turn_left():
    if robot:
        robot.left(0.3)
        time.sleep(1)
        robot.stop()

def turn_right():
    if robot:
        robot.right(0.3)
        time.sleep(1)
        robot.stop()

def stop():
    if robot:
        robot.stop()

def set_speed_left(speed):
    if robot:
        robot.left_motor.value = speed

def set_speed_right(speed):
    if robot:
        robot.right_motor.value = speed

def update_camera():
    if camera:
        # Assuming the camera.value is a numpy array in BGR format
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(camera.value, cv2.COLOR_BGR2RGB)
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(rgb_image)
        return pil_image
    else:
        # Load the placeholder image as a PIL Image
        with open(placeholder_image_path, 'rb') as f:
            pil_image = Image.open(f)
            pil_image.load()  # This might be necessary depending on how PIL handles lazy loading
        return pil_image


def save_snapshot():
    if camera:
        image_path = f"snapshots/{uuid.uuid4()}.jpeg"
        with open(image_path, 'wb') as f:
            f.write(update_camera())
        return image_path
    else:
        return placeholder_image_path

# Ensure snapshot directory exists
os.makedirs('snapshots', exist_ok=True)

with gr.Blocks() as demo:
    gr.Markdown("# JetBot Control Panel")
    gr.Markdown("Use the controls below to operate the JetBot. Ensure the area is clear before sending commands to avoid collisions.")
    
    with gr.Row():
        left_speed = gr.Slider(-1.0, 1.0, step=0.1, label="Left Motor Speed", value=0.0)
        right_speed = gr.Slider(-1.0, 1.0, step=0.1, label="Right Motor Speed", value=0.0)
        
    with gr.Row():
        gr.Button("⬆️ Forward", elem_id="forward_button").click(move_forward)
        gr.Button("⬅️ Left", elem_id="left_button").click(turn_left)
        gr.Button("🛑 Stop", elem_id="stop_button").click(stop)
        gr.Button("➡️ Right", elem_id="right_button").click(turn_right)
        gr.Button("⬇️ Backward", elem_id="backward_button").click(move_backward)
    
    left_speed.change(set_speed_left, inputs=[left_speed], outputs=[])
    right_speed.change(set_speed_right, inputs=[right_speed], outputs=[])

    with gr.Row():
        live_feed = gr.Image(streaming=True,value=update_camera)
        snapshot_result = gr.Image(width=300, height=300, label="Last Snapshot")
    snapshot_button = gr.Button("📸 Take Snapshot", elem_id="snapshot_button").click(save_snapshot, outputs=snapshot_result)
        
    gr.Markdown("### Instructions")
    gr.Markdown("1. **Move the Sliders**: Adjust the sliders to change the speed of the left and right motors.")
    gr.Markdown("2. **Press the Buttons**: Use the directional buttons to move the JetBot. Press 'Stop' to halt any movement.")
    gr.Markdown("3. **Camera and Snapshot**: View the live feed and take snapshots using the button.")

demo.launch()
