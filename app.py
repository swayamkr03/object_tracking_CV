import gradio as gr
from main import process_video
import shutil
import os

def run(video):
    if video is None:
        return None

    os.makedirs("data", exist_ok=True)

    input_path = "data/temp_input.mp4"
    shutil.copy(video, input_path)

    output = process_video(input_path)
    return output


gr.Interface(
    fn=run,
    inputs=gr.Video(label="Upload Surveillance Video"),
    outputs=gr.File(label="Processed Video"),
    title="Smart Surveillance System",
    description="Vehicle detection and tracking using YOLOv8 + DeepSORT"
).launch()