# KavachX Edge: fire/smoke, motion and intrusion detection (QCS6490)

Three detections built for the KavachX Edge Box (Qualcomm QCS6490), implemented using pre-trained models and tested on an Apple Silicon development host and Qualcomm AI Hub dev kits (physical Edge Box deployment pending hardware delivery):

> 📄 **Technical Submission Report**: For the complete engineering write-up, bare-metal hardware benchmark data, Qualcomm AI Hub job links, and capacity formulas, see [**docs/SUBMISSION_REPORT.md**](docs/SUBMISSION_REPORT.md).

| Detection | Method |
|---|---|
| Fire & smoke | YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`, `best.pt`; class 0 smoke, 1 fire) |
| Motion | OpenCV `BackgroundSubtractorMOG2` on a 320 px downscaled frame |
| Intrusion | COCO YOLOv8n, class 0 (person), foot point inside a restricted-zone polygon |

**Design for capacity & stationary intruders:** MOG2 runs on every frame (~1.1 ms on a 320 px downscaled frame). Person detection runs every N frames during motion, continues for `--hangover-seconds` (default 3.0s) after motion ceases, and runs periodically every `--keepalive-seconds` (default 1.0s) even with zero motion so an intruder who stops moving inside a restricted zone is not lost. The fire/smoke model runs on a fixed interval because smoke plumes can expand slowly without triggering fast pixel motion.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # use opencv-python-headless on the Edge Box
python scripts/download_models.py          # fills ./models
python scripts/make_test_video.py test.mp4 # synthetic test clip, or supply your own camera clip
```

## Run
```bash
# Production gated pipeline (with zone overlay and stationary intruder protection)
python -m src.pipeline --source videos/person_walk.mp4 \
  --zone "0.3,0.4;0.9,0.4;0.9,1.0;0.3,1.0" \
  --warmup-seconds 2.0 --hangover-seconds 3.0 --keepalive-seconds 1.0 \
  --person-interval 3 --fire-interval 10

# Baseline without gating (for comparison)
python -m src.pipeline --source videos/person_walk.mp4 --no-gate --person-interval 1

# Key pipeline flags:
#   --warmup-seconds FLOAT     MOG2 background calibration in seconds (default: 2.0s; auto-scales to video FPS)
#   --hangover-seconds FLOAT   Keep running person checks for N seconds after motion stops (default: 3.0s)
#   --keepalive-seconds FLOAT  Run periodic person check even during zero motion (default: 1.0s; 0 to disable)
#   --person-interval INT      Evaluate person detector every N frames during active motion (default: 3)
#   --fire-interval INT        Evaluate fire/smoke detector every N frames (default: 10)
#   --zone STR                 Normalized polygon coordinates "x1,y1;x2,y2;..."

# Per-model benchmarks -> results/benchmarks.csv
python -m src.benchmark --mode motion --source test.mp4 --label mog2
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source test.mp4 --label person_fp32
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source test.mp4 --label fire_fp32
```

## Repository Media & Proof Artifacts

All raw input videos, annotated pipeline outputs, and visual proofs are included directly in this repository:

| Asset | Description | File Path |
|---|---|---|
| **Intrusion Annotated Video** | Annotated detection output demonstrating bounding boxes and red `[ZONE!]` intrusion alert at 13.00s | [`outputs/person_walk_annotated.mp4`](outputs/person_walk_annotated.mp4) |
| **Fire/Smoke Annotated Video** | Annotated detection output demonstrating smoke bounding boxes across all periodic checks | [`outputs/fire_smoke_annotated.mp4`](outputs/fire_smoke_annotated.mp4) |
| **Zone Preview Overlay** | Image preview of the virtual restricted-zone polygon overlaid on the camera scene | [`outputs/zone_preview.jpg`](outputs/zone_preview.jpg) |
| **Raw Walking Clip** | Input 60-frame video (2.0 FPS, 30.0s) used for motion & intrusion evaluation | [`videos/person_walk.mp4`](videos/person_walk.mp4) |
| **Raw Fire/Smoke Clip** | Input 60-frame video (2.0 FPS, 30.0s) used for fire/smoke evaluation | [`videos/fire_smoke.mp4`](videos/fire_smoke.mp4) |
| **D-Fire Sample Images** | Sample images extracted from the D-Fire dataset for inspection | [`data/dfire_samples/`](data/dfire_samples/) |
| **YOLO Prediction Runs** | Sample detection output directories from validation runs | [`runs/detect/`](runs/detect/) |
| **Synthetic Test Video** | 200-frame synthetic clip used for initial pipeline benchmarking and AI Hub calibration | [`test.mp4`](test.mp4) |

## INT8 / NPU
```bash
python scripts/export_int8.py --model models/yolov8n.pt --data coco8.yaml
python scripts/export_int8.py --model models/fire-smoke-best.pt --data <dfire.yaml>
```
On Qualcomm AI Hub, models were quantized (INT8 post-training quantization) and compiled into QNN context binaries targeting the Hexagon NPU on the QCS6490 chipset.

## Results
*All numbers are measured and labelled by source:*
*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is isolated model inference peak memory (`estimated_inference_peak_memory`).*

| Config | Source | mean ms | FPS | P95 ms | Peak RAM* | CPU % (Mac)* | Compute Units / Notes | Job Reference |
|---|---|---|---|---|---|---|---|---|
| **Person FP32 (QNN)** | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 MB | - | GPU / CPU fallback (224 GPU, 15 CPU, 13 NPU ops) | [`jgjyw1jvg`](https://workbench.aihub.qualcomm.com/jobs/jgjyw1jvg/) |
| **Person INT8 (NPU)** | **AI Hub QCS6490** | **8.13** | **123.0** | **8.84** | **16.6 MB** | - | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** | [`j5mkdordp`](https://workbench.aihub.qualcomm.com/jobs/j5mkdordp/) |
| **Fire/smoke FP32 (QNN)**| AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 MB | - | GPU / CPU fallback (222 GPU, 15 CPU, 13 NPU ops) | [`jgzy4w1op`](https://workbench.aihub.qualcomm.com/jobs/jgzy4w1op/) |
| **Fire/smoke INT8 (NPU)**| **AI Hub QCS6490** | **6.47** | **154.5** | **7.17** | **13.6 MB** | - | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** | [`jgnq7o2kg`](https://workbench.aihub.qualcomm.com/jobs/jgnq7o2kg/) |
| **MOG2 (Stock Clip, person_walk)** | Mac M1 | 1.14 | 874.0 | 1.48 | 714 MB | 99.0% (1 core) | Single CPU core (320 px downscaled, stock clip) | `results/benchmarks.csv` |
| **Person FP32 (Stock Clip, person_walk)** | Mac M1 | 34.92 | 28.6 | 39.34 | 1,008 MB | 196.5% (~2 cores) | PyTorch CPU runtime (imgsz 640, stock clip) | `results/benchmarks.csv` |
| **Fire/Smoke FP32 (Stock Clip, fire_smoke)** | Mac M1 | 33.61 | 29.8 | 40.64 | 1,044 MB | 194.9% (~2 cores) | PyTorch CPU runtime (imgsz 640, stock clip) | `results/benchmarks.csv` |
| **MOG2 (Stock Clip, fire_smoke)** | Mac M1 | 1.13 | 887.0 | 1.64 | 715 MB | 97.8% (1 core) | Single CPU core (320 px downscaled, stock clip) | `results/benchmarks.csv` |

### Pipeline Gating & Event Evidence (`videos/person_walk.mp4`, 60 frames, 2.0 FPS, 30.0s)
*Wall times (1–3 seconds) on short 60-frame clips are dominated by PyTorch import, model loading, and startup overhead rather than steady-state throughput; comparisons below report exact model invocation call counts.*

| Configuration | Parameters | Person Model Calls | Detection & Intrusion Behavior |
|---|---|---|---|
| **Pure Gated** | `--person-interval 3 --hangover-seconds 0 --keepalive-seconds 0` | **4 calls** | Skips static frames (4 calls vs 20 baseline). **Vulnerability:** Misses intruder who stops moving before/at the zone (0 intrusion alerts). |
| **Production Gated** | `--person-interval 3 --hangover-seconds 3.0 --keepalive-seconds 1.0` | **34 calls** | **Catches stationary intruder:** Intrusion alert fires at **13.00s (frame 26)** and continues detecting intruder. |
| **Ungated Interval 3** | `--no-gate --person-interval 3` | **20 calls** | Evaluates fixed 1-in-3 frames regardless of motion ($60 / 3 = 20$). |
| **Continuous Baseline** | `--no-gate --person-interval 1` | **60 calls** | Evaluates every frame indiscriminately. |

> **Frame Rate Adaptation Note:** On this 2.0 FPS test clip, the 1.0s keepalive evaluates every 2 frames ($1.0 \times 2 = 2$). This is more frequent than interval 3 ($1.5\text{s}$), which is why production gating executes 34 calls on this specific low-FPS clip. On a standard **30 FPS surveillance camera**, keepalive is 1 call/s while interval 3 is 10 calls/s—achieving a **90% call reduction** during idle periods while guaranteeing stationary intruder detection.
>
> **Detection & Negative-Clip Evidence:** On `videos/fire_smoke.mp4` (60 frames, 2.0 FPS), smoke was detected at all six periodic checks (5.0s, 10.0s, 15.0s, 20.0s, 25.0s, and 30.0s). In negative-control testing on these two 60-frame clips, no false alerts were observed (0 fire alerts on the walking clip across all 6 checks, and 0 person alerts on the smoke clip across all 60 frames). True false-alarm rates will require extensive validation across larger negative datasets.

### Camera Capacity Analysis (Estimate, Not a Result)
*Results from Qualcomm AI Hub measure isolated model inference on a Dragonwing RB3 Gen 2 dev kit, not end-to-end box performance. On-device RTSP video decoding (H.264/H.265), frame resizing/color conversion, NMS post-processing, and OpenCV MOG2 on the QCS6490 CPU are currently unmeasured.*

1. **NPU-Bound Budget (Inference Only)**:
   - **Assumptions**: 30 FPS camera feeds.
     - Person model (INT8, 8.13 ms): evaluated every 3rd frame during active motion (10 calls/s) plus a 1-second keepalive during idle periods (1 call/s).
       $$\text{Person calls/s} = 10 \cdot f + 1 \cdot (1 - f)$$
       where $f$ is the active motion share (including hangover). Note: $f$ is an operational assumption, not a measurement.
     - Fire/smoke model (INT8, 6.47 ms): evaluated on a fixed interval of every 10 frames (3 calls/s) $\rightarrow 3 \times 6.47 \text{ ms} = \mathbf{19.41 \text{ ms/s}}$.
     - Safe duty cycle: **70% NPU utilization limit** (700 ms/s available compute budget) to prevent thermal throttling and queuing delay.

   - **Scenario A: 25% Active Motion ($f = 0.25$)**:
     - Person calls/s: $10 \times 0.25 + 1 \times 0.75 = 2.5 + 0.75 = \mathbf{3.25 \text{ calls/s}}$
     - Person NPU time: $3.25 \times 8.13 \text{ ms} = \mathbf{26.42 \text{ ms/s}}$
     - Total NPU time/s per camera: $26.42 \text{ ms} + 19.41 \text{ ms} = \mathbf{45.83 \text{ ms/s}}$
     - Estimated NPU capacity at 70% load: $700 \text{ ms} / 45.83 \text{ ms} \approx \mathbf{15.27} \rightarrow \mathbf{\sim 15 \text{ cameras}}$.

   - **Scenario B: 40% Active Motion ($f = 0.40$)**:
     - Person calls/s: $10 \times 0.40 + 1 \times 0.60 = 4.0 + 0.60 = \mathbf{4.60 \text{ calls/s}}$
     - Person NPU time: $4.60 \times 8.13 \text{ ms} = \mathbf{37.40 \text{ ms/s}}$
     - Total NPU time/s per camera: $37.40 \text{ ms} + 19.41 \text{ ms} = \mathbf{56.81 \text{ ms/s}}$
     - Estimated NPU capacity at 70% load: $700 \text{ ms} / 56.81 \text{ ms} \approx \mathbf{12.32} \rightarrow \mathbf{\sim 12 \text{ cameras}}$.

   - **Estimated NPU capacity range**: **~12 to 15 cameras at 30 FPS** (inference-only, 70% NPU load).

2. **CPU-Bound Budget (Decode, Pre/Post-processing, MOG2)**:
   - On Mac M1, MOG2 (320 px) takes **1.14 ms/frame** on stock footage (~34.2 ms/s @ 30 FPS on 1 core).
   - On the Edge Box, handling 12–15 concurrent RTSP decode pipelines and OpenCV operations on the Kryo CPU may become the primary bottleneck before NPU saturation. Hardware video decoding (via GStreamer / Qualcomm multimedia accelerator `v4l2h264dec`) is recommended.

## Known Limitations & Production Gaps
- **POC Evaluation Scale**: Validated on individual royalty-free stock test clips (Mixkit/Pexels). Full accuracy evaluation requires benchmark test sets across diverse lighting, weather, and camera angles.
- **Alert De-Duplication**: Alerts fire at each detection interval without stateful tracking. A commercial deployment requires an object tracker (e.g., ByteTrack) and temporal hysteresis.
- **Fire/Smoke Recall**: D-Fire test split published recall is ~0.688 and mAP50-95 is 0.362. Missed detections in complex lighting are expected.
- **Licensing**: Ultralytics YOLOv8 models are AGPL-3.0. Commercial deployments require an Ultralytics enterprise licence or an alternative model with a verified permissive licence (e.g., Apache-2.0 or BSD).
- **INT8 Accuracy Unverified**: INT8 quantization was performed on Qualcomm AI Hub using 10 synthetic calibration frames from `test.mp4`. While this confirms 100% NPU operator mapping and latency, mAP accuracy on validation splits remains unmeasured.
- **Edge Box Physical Deployment**: Full end-to-end pipeline execution with hardware video decode on the physical KavachX Edge Box remains pending hardware access.
