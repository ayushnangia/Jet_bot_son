import gradio as gr
import os
import uuid
from PIL import Image
import numpy as np
import cv2
import time

# Placeholder image path or URL
placeholder_image_path = 'images.png'

# Ensure snapshot directory exists
os.makedirs('snapshots', exist_ok=True)

try:
    from jetbot import Robot, Camera
    robot = Robot()
    camera = Camera.instance(width=224, height=224)  # Adjust resolution if necessary
except ImportError as e:
    print(f"Import error: {e}. Running in simulation mode.")
    robot = None
    camera = None

# Globals for recording and command log
is_recording = False
command_log = []

def toggle_recording():
    global is_recording
    is_recording = not is_recording
    if not is_recording:
        return "Recording stopped. Ready to replay."
    else:
        command_log.clear()
        return "Recording started. Please execute commands."

def record_command(command, *params):
    if is_recording:
        command_log.append((command, params))

def replay_commands():
    if not command_log:
        return "No commands recorded."
    for func, params in command_log:
        func(*params)
        time.sleep(1)  # delay between commands for clarity
    return "Replay finished."

def move_robot(direction, duration=1.0):
    if robot:
        getattr(robot, direction)(0.3)
        robot.sleep(duration)
        robot.stop()
    record_command(move_robot, direction, duration)

def set_motor_speed(left_speed, right_speed):
    if robot:
        robot.left_motor.value = left_speed
        robot.right_motor.value = right_speed
    record_command(set_motor_speed, left_speed, right_speed)

def get_camera_image():
    if camera:
        return cv2.cvtColor(camera.value, cv2.COLOR_BGR2RGB)
    else:
        return Image.open(placeholder_image_path)

def update_live_feed(image_component):
    new_image = get_camera_image()
    image_component.update(new_image)

def save_snapshot():
    image = get_camera_image()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image_path = f'snapshots/{uuid.uuid4()}.jpeg'
    image.save(image_path, 'JPEG')
    return image_path

with gr.Blocks() as demo:
    gr.Markdown("# JetBot Control Panel")

    with gr.Row():
        record_button = gr.Button("🔴 Start/⏹️ Stop Recording")
        replay_button = gr.Button("🔁Replay Commands")
        record_status = gr.Textbox(label="Recording Status", value="Recording not started.")
    
    record_button.click(toggle_recording, outputs=record_status)
    replay_button.click(replay_commands, outputs=record_status)

    with gr.Row():
        left_speed = gr.Slider(-1.0, 1.0, step=0.1, label="Left Motor Speed", value=0.0)
        right_speed = gr.Slider(-1.0, 1.0, step=0.1, label="Right Motor Speed", value=0.0)
        left_speed.change(fn=set_motor_speed, inputs=[left_speed, right_speed], outputs=[])
        right_speed.change(fn=set_motor_speed, inputs=[left_speed, right_speed], outputs=[])

    with gr.Row():
        directions = ['forward', 'left', 'stop', 'right', 'backward']
        for direction in directions:
            gr.Button(direction.title(), elem_id=f"{direction}_button").click(
                fn=lambda x=direction: move_robot(x), inputs=[], outputs=[])

    with gr.Row():
        live_feed = gr.Image()
        snapshot_button = gr.Button("📸 Take Snapshot")
        snapshot_result = gr.Image(label="Last Snapshot")
        snapshot_button.click(fn=save_snapshot, outputs=[snapshot_result])

    gr.Markdown("### Instructions")
    gr.Markdown("1. **Move the Sliders**: Adjust the sliders to change the speed of the left and right motors.")
    gr.Markdown("2. **Press the Buttons**: Use the directional buttons to move the JetBot. Press 'Stop' to halt any movement.")
    gr.Markdown("3. **Camera and Snapshot**: View the updated live feed and take snapshots using the button.")
    gr.Markdown("4. **Recording and Replay**: Start recording commands by pressing 'Start/Stop Recording', and replay them with 'Replay Commands'.")

demo.launch()
