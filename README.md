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
*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is isolated model inference peak memory (`estimated_inference_peak_memory`).*

| Config | Source | mean ms | FPS | P95 ms | Peak RAM* | CPU % (Mac)* | Compute Units / Notes | Job Reference |
|---|---|---|---|---|---|---|---|---|
| **Person FP32 (QNN)** | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 MB | - | GPU / CPU fallback (224 GPU, 15 CPU, 13 NPU ops) | [`jgjyw1jvg`](https://workbench.aihub.qualcomm.com/jobs/jgjyw1jvg/) |
| **Person INT8 (NPU)** | **AI Hub QCS6490** | **8.13** | **123.0** | **8.84** | **16.6 MB** | - | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** | [`j5mkdordp`](https://workbench.aihub.qualcomm.com/jobs/j5mkdordp/) |
| **Fire/smoke FP32 (QNN)**| AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 MB | - | GPU / CPU fallback (222 GPU, 15 CPU, 13 NPU ops) | [`jgzy4w1op`](https://workbench.aihub.qualcomm.com/jobs/jgzy4w1op/) |
| **Fire/smoke INT8 (NPU)**| **AI Hub QCS6490** | **6.47** | **154.5** | **7.17** | **13.6 MB** | - | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** | [`jgnq7o2kg`](https://workbench.aihub.qualcomm.com/jobs/jgnq7o2kg/) |
| **MOG2 (Real Clip, person_walk)** | Mac M1 | 1.14 | 874.0 | 1.48 | 714 MB | 99.0% (1 core) | Single CPU core (320 px downscaled, real clip) | `results/benchmarks.csv` |
| **Person FP32 (Real Clip, person_walk)** | Mac M1 | 34.92 | 28.6 | 39.34 | 1,008 MB | 196.5% (~2 cores) | PyTorch CPU runtime (imgsz 640, real clip) | `results/benchmarks.csv` |
| **Fire/Smoke FP32 (Real Clip, fire_smoke)** | Mac M1 | 33.61 | 29.8 | 40.64 | 1,044 MB | 194.9% (~2 cores) | PyTorch CPU runtime (imgsz 640, real clip) | `results/benchmarks.csv` |
| **MOG2 (Real Clip, fire_smoke)** | Mac M1 | 1.13 | 887.0 | 1.64 | 715 MB | 97.8% (1 core) | Single CPU core (320 px downscaled, real clip) | `results/benchmarks.csv` |

### Gating Impact on Real Clip (`person_walk.mp4`)
Under fair baseline comparison with identical frame-skip (`--person-interval 3`):
* **Without Gating (`--no-gate`)**: Evaluated **20 person model calls** (34.8 FPS).
* **With MOG2 Gating**: Evaluated **4 person model calls** (37.4 FPS).
* **Benefit**: Gating eliminated **80% of deep learning invocations** on the walking clip while retaining full alert coverage. Against continuous per-frame inference (60 calls, 14.0 FPS), gating delivers a **2.67x speedup**.

### Camera Capacity Analysis (Estimate, Not a Result)
*Results from Qualcomm AI Hub measure isolated model inference on a Dragonwing RB3 Gen 2 dev kit, not end-to-end box performance. On-device RTSP video decoding (H.264/H.265), frame resizing/color conversion, NMS post-processing, and OpenCV MOG2 on the QCS6490 CPU are currently unmeasured.*

1. **NPU-Bound Budget (Inference Only)**:
   - **Person Detection (INT8)**: 8.13 ms per call. Evaluated every 3rd frame during motion (~25% active motion $\approx$ 2.5 inf/s) $\rightarrow \mathbf{20.3 \text{ ms/s}}$.
   - **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames (~3 inf/s) $\rightarrow \mathbf{19.4 \text{ ms/s}}$.
   - **Total NPU inference budget per 30 FPS camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/s}}$.
   - **Estimated NPU capacity at safe 70% NPU duty cycle**: $700 \text{ ms} / 39.7 \text{ ms} \approx \mathbf{\sim 17 \text{ cameras}}$. At higher motion activity (~40% motion $\rightarrow 52.0 \text{ ms/s}$ load), capacity is $700 / 52 \approx \mathbf{\sim 13 \text{ cameras}}$.
   - **Estimated capacity range**: **~13 to 17 cameras (at 70% NPU load)**.

2. **CPU-Bound Budget (Decode, Pre/Post-processing, MOG2)**:
   - On Mac M1, MOG2 (320 px) takes **1.14 ms/frame** on real footage (~34.2 ms/s @ 30 FPS on 1 core).
   - On the Edge Box, handling 13–17 concurrent RTSP decode pipelines and OpenCV operations on the Kryo CPU may become the primary bottleneck before NPU saturation. Hardware video decoding (via GStreamer / Qualcomm multimedia accelerator `v4l2h264dec`) is recommended.

## Known Limitations & Production Gaps
- **POC Evaluation Scale**: Validated on individual test clips. Full accuracy evaluation requires benchmark test sets across diverse lighting, weather, and camera angles.
- **Alert De-Duplication**: Alerts fire at each detection interval without stateful tracking. A commercial deployment requires an object tracker (e.g., ByteTrack) and temporal hysteresis.
- **Fire/Smoke Recall**: D-Fire test split published recall is ~0.688 and mAP50-95 is 0.362. Missed detections in complex lighting are expected.
- **Licensing**: Ultralytics YOLOv8 models are AGPL-3.0. Check licensing terms before commercial deployment.
- **Edge Box Physical Deployment**: Full end-to-end pipeline execution with hardware video decode on the physical KavachX Edge Box remains pending hardware access.
