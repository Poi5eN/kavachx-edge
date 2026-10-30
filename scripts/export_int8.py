"""Export YOLOv8n to INT8 TFLite (step 1 of the NPU path).

Usage
  python scripts/export_int8.py --model models/yolov8n.pt --data coco8.yaml
  python scripts/export_int8.py --model models/fire-smoke-best.pt --data path/to/dfire.yaml

IMPORTANT: calibration data must match the model's domain. Use a few hundred
D-Fire images for the fire/smoke model, not COCO, or INT8 accuracy can collapse.

Running on the QCS6490 Hexagon NPU additionally needs the Qualcomm toolchain.
Two routes: (a) Qualcomm AI Hub compile/quantize job for the target device, or
(b) load the INT8 .tflite through LiteRT with the QNN delegate on the board.
Verify the exact SDK version installed on the Edge Box before choosing.
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True, help="dataset yaml used for INT8 calibration")
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

path = YOLO(a.model).export(format="tflite", int8=True, data=a.data, imgsz=a.imgsz)
print("exported:", path)
