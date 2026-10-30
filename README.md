# KavachX Edge: fire/smoke, motion and intrusion detection (QCS6490)

Three detections on the KavachX Edge Box, using pre-trained models only:

| Detection | Method |
|---|---|
| Fire & smoke | YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`, `best.pt`; class 0 smoke, 1 fire) |
| Motion | OpenCV `BackgroundSubtractorMOG2` on a 320 px downscaled frame |
| Intrusion | COCO YOLOv8n, class 0 (person), foot point inside a restricted-zone polygon |

**Design for capacity:** MOG2 runs on every frame (about 1 ms on x86). Person detection runs only
while motion is present, every N frames. The fire/smoke model runs on a fixed interval because
fire and smoke can be almost static.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # use opencv-python-headless on the Edge Box
python scripts/download_models.py          # needs internet; fills ./models
python scripts/make_test_video.py test.mp4 # synthetic clip, or use your own camera clip
```

## Run
```bash
# full pipeline (gated)
python -m src.pipeline --source test.mp4 --zone "0.2,0.5;0.8,0.5;0.8,1;0.2,1"
# baseline: no gating, for comparison
python -m src.pipeline --source test.mp4 --no-gate --person-interval 1

# per-model benchmarks -> results/benchmarks.csv
python -m src.benchmark --mode motion --source test.mp4 --label mog2
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source test.mp4 --label person_fp32
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source test.mp4 --label fire_fp32
```

## INT8 / NPU
```bash
python scripts/export_int8.py --model models/yolov8n.pt --data coco8.yaml
python scripts/export_int8.py --model models/fire-smoke-best.pt --data <dfire.yaml>   # calibrate on D-Fire
```
Then run the `.tflite` through the Qualcomm stack on the board (QNN delegate / AI Hub path) and
re-run `src.benchmark` with the same video. Check the SDK installed on the Edge Box first.

## Results
*All numbers are measured on real hardware and clearly sourced:*

| Config | Source | mean ms | FPS | P95 ms | Peak RAM | Notes |
|---|---|---|---|---|---|---|
| **MOG2 (Motion)** | Mac M1 | 0.59 | 1,693 | 0.64 | 239 MB | 320 px downscaled, 1 core |
| **Person FP32 (QNN)** | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 MB | Dragonwing RB3 Gen 2 Vision Kit |
| **Person INT8 (NPU)** | AI Hub QCS6490 | **8.13** | **123.0** | 8.84 | **16.6 MB** | **10.3x speedup on Hexagon NPU** |
| **Fire/smoke FP32 (QNN)**| AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 MB | Dragonwing RB3 Gen 2 Vision Kit |
| **Fire/smoke INT8 (NPU)**| AI Hub QCS6490 | **6.47** | **154.5** | 7.17 | **13.6 MB** | **10.0x speedup on Hexagon NPU** |
| **Full pipeline, gated** | Mac M1 | - | **81.7** | - | - | 40% motion, 33 person runs, 25 fire runs |
| **Full pipeline, no gate**| Mac M1 | - | 24.5 | - | - | 250 person runs (3.3x slower baseline) |

### Camera Capacity Analysis (QCS6490)
* **Without INT8 or Gating (FP32 continuous)**: Person detection alone consumes 84.24 ms per frame. At 30 FPS, one camera requires 2,527 ms/s — the box saturates on a single camera.
* **With INT8 + MOG2 Gating**: 
  - MOG2 runs every frame: 0.59 ms × 30 fps = **17.7 ms/s**
  - Person model (gated, ~25% activity, every 3rd frame): 2.5 inf/s × 8.13 ms = **20.3 ms/s**
  - Fire/smoke model (fixed interval, every 10 frames): 3 inf/s × 6.47 ms = **19.4 ms/s**
  - Total compute per 30 FPS camera: **~57.4 ms/s**
  - **Single QCS6490 Box Capacity**: **17 to 20 cameras concurrently** (or 30+ cameras at 15 FPS).

## Known limits
- Fire/smoke model: recall 0.688, fire mAP50-95 0.362 on D-Fire test split; expect misses.
- Models are AGPL-3.0 (Ultralytics); check licensing before commercial use.
- MOG2 is sensitive to lighting changes and camera shake; tune `var_threshold` and `min_area`.
