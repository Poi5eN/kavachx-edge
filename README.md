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

## Results (fill in from the Edge Box, not from a laptop)
| Config | mean ms | FPS | CPU % | Notes |
|---|---|---|---|---|
| MOG2 | | | | |
| Person FP32 CPU | | | | |
| Person INT8 (NPU) | | | | |
| Fire/smoke FP32 CPU | | | | |
| Fire/smoke INT8 (NPU) | | | | |
| Full pipeline, gated | | | | |
| Full pipeline, no gate | | | | |

## Known limits
- Fire/smoke model: recall 0.688, fire mAP50-95 0.362 on D-Fire test split; expect misses.
- Models are AGPL-3.0 (Ultralytics); check licensing before commercial use.
- MOG2 is sensitive to lighting changes and camera shake; tune `var_threshold` and `min_area`.
