"""Download the pre-trained weights into ./models (run on a machine with internet)."""
import os
import shutil

os.makedirs("models", exist_ok=True)

from huggingface_hub import hf_hub_download
p = hf_hub_download("rabahdev/fire-smoke-yolov8n", "best.pt")
shutil.copy(p, "models/fire-smoke-best.pt")
print("fire/smoke ->", "models/fire-smoke-best.pt")

from ultralytics import YOLO
YOLO("yolov8n.pt")                       # auto-downloads to cwd
if os.path.exists("yolov8n.pt"):
    shutil.move("yolov8n.pt", "models/yolov8n.pt")
print("person      -> models/yolov8n.pt")
