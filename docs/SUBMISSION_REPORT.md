# KavachX Edge Detection System — Technical Submission Report

**Candidate:** Gourav Kumar Upadhyay  
**Date:** 2026-10-31  
**GitHub Repository:** [https://github.com/Poi5eN/kavachx-edge](https://github.com/Poi5eN/kavachx-edge)  
**Target Hardware:** Qualcomm QCS6490 ("Edge Box" / Dragonwing RB3 Gen 2 Vision Kit, Hexagon v68 NPU)  

---

## 1. Executive Summary & Compliance Verdict

All three required computer vision detections have been implemented and functionally validated on royalty-free stock video clips (Mixkit/Pexels). Deep learning model inference has been benchmarked directly on Qualcomm QCS6490 hardware at the model level via Qualcomm AI Hub. Full end-to-end pipeline execution on the physical KavachX Edge Box and on-device hardware utilization monitoring remain pending hardware access.

| Assignment Requirement | Implementation / Methodology | Verification Status | Key Result & Evidence |
|---|---|---|---|
| **1. Fire & Smoke Detection** | YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`). Class 0: `smoke`, Class 1: `fire`. Rate-limited every 10 frames. | **Met on stock clip** | Validated on `videos/fire_smoke.mp4`: smoke alerts fired every 10 frames starting from **5.00s**. INT8 inference takes **6.47 ms** on QCS6490 NPU. |
| **2. Motion Detection** | OpenCV `BackgroundSubtractorMOG2` downscaled to 320 px width with morphological opening. | **Met on stock clip** | Validated on `videos/person_walk.mp4`: runs in **1.14 ms** per frame (>870 FPS) on Mac M1 (1 CPU core). |
| **3. Intrusion Detection** | Pre-trained COCO YOLOv8n (class 0: `person`). Bounding box foot-point tracked inside a user-defined normalized polygon. | **Met on stock clip** | Validated on `videos/person_walk.mp4`: intrusion alert fired at **13.00s** when foot entered zone. INT8 inference takes **8.13 ms** on QCS6490 NPU. |
| **4. Direct Testing on QCS6490** | Compiled and profiled bare-metal inference jobs on Qualcomm AI Hub cloud testbed targeting the **Dragonwing RB3 Gen 2 Vision Kit (QCS6490 chipset)**. | **Partially Met** | Tested model inference across FP32 and INT8 precisions. Full pipeline with RTSP video decode on physical KavachX Edge Box is pending. |
| **5. Integrate OpenCV for Motion** | Architected motion-gated detection pipeline in `src/pipeline.py` with adaptive warmup, motion hangover, and periodic keepalive. | **Met in pipeline** | Production gating (`hangover=3s`, `keepalive=1s`) catches stationary intruders (intrusion alert at **13.00s**, 34 calls on 2 FPS clip). Pure gating evaluates **4 calls** but misses stationary targets. |
| **6. Explore INT8 Quantization** | Applied post-training INT8 quantization (PTQ) targeting Hexagon v68 NPU vector/tensor accelerators (HTP/HVX). | **Explored, accuracy unverified** | Verified on QCS6490 NPU: **8.13 ms** (Person) and **6.47 ms** (Fire) with 100% of operators compiled to NPU. Calibration used 10 synthetic frames; mAP validation is pending. |
| **7. Measure Inference Speed & CPU/NPU Utilization** | Benchmarked latency, P95, throughput, compute-unit mapping, and RAM. | **Speed met; utilization not met** | AI Hub provides execution timing, peak memory, and operator compute-unit mapping, but does not report percentage NPU/CPU utilization. |
| **8. Maximize Camera Capacity & Minimize Resources** | Formulated an isolated NPU vs CPU budget model with safety margins. | **Estimate only** | **~12 to 15 concurrent 30 FPS cameras per box** at a safe 70% NPU duty cycle (assuming 25%–40% active motion; on-device CPU decoding unmeasured). |

---

## 2. Hardware Benchmark Proof (Qualcomm QCS6490 & Mac M1)

Inference was profiled directly on physical **Qualcomm QCS6490 hardware** via Qualcomm AI Hub testbeds (`Dragonwing RB3 Gen 2 Vision Kit`, Hexagon v68 NPU):

*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is isolated model inference peak memory (`estimated_inference_peak_memory`). AI Hub links require a Qualcomm AI Hub login.*

| Model Configuration | Target Device | Precision | Mean Latency | Throughput | P95 Latency | Peak RAM* | CPU % (Mac)* | Compute Unit Operator Breakdown | AI Hub Job Reference |
|---|---|---|---|---|---|---|---|---|---|
| **Person Detection** (COCO YOLOv8n) | AI Hub QCS6490 | FP32 | 84.24 ms | 11.9 FPS | 85.92 ms | 45.3 MB | - | 224 GPU, 15 CPU, 13 NPU ops (GPU fallback) | Profile: [`jgjyw1jvg`](https://workbench.aihub.qualcomm.com/jobs/jgjyw1jvg/)<br>Compile: [`jgl817q8p`](https://workbench.aihub.qualcomm.com/jobs/jgl817q8p/) |
| **Person Detection** (COCO YOLOv8n) | **AI Hub QCS6490** | **INT8** | **8.13 ms** | **123.0 FPS** | **8.84 ms** | **16.6 MB** | - | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** | Profile: [`j5mkdordp`](https://workbench.aihub.qualcomm.com/jobs/j5mkdordp/)<br>Compile: [`jgl81ojjp`](https://workbench.aihub.qualcomm.com/jobs/jgl81ojjp/)<br>Quant: [`j57n7kqvp`](https://workbench.aihub.qualcomm.com/jobs/j57n7kqvp/) |
| **Fire/Smoke Detection** (`best.pt`) | AI Hub QCS6490 | FP32 | 64.87 ms | 15.4 FPS | 66.35 ms | 39.7 MB | - | 222 GPU, 15 CPU, 13 NPU ops (GPU fallback) | Profile: [`jgzy4w1op`](https://workbench.aihub.qualcomm.com/jobs/jgzy4w1op/)<br>Compile: [`jgd4z0yrg`](https://workbench.aihub.qualcomm.com/jobs/jgd4z0yrg/) |
| **Fire/Smoke Detection** (`best.pt`) | **AI Hub QCS6490** | **INT8** | **6.47 ms** | **154.5 FPS** | **7.17 ms** | **13.6 MB** | - | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** | Profile: [`jgnq7o2kg`](https://workbench.aihub.qualcomm.com/jobs/jgnq7o2kg/)<br>Compile: [`j56mdrk6p`](https://workbench.aihub.qualcomm.com/jobs/j56mdrk6p/)<br>Quant: [`jp449mz8p`](https://workbench.aihub.qualcomm.com/jobs/jp449mz8p/) |
| **MOG2 (Stock Clip, `person_walk`)** | Mac M1 | - | 1.14 ms | 874.0 FPS | 1.48 ms | 714 MB | 99.0% (1 core) | Single CPU core (320 px downscaled, stock clip) | Logged in `benchmarks.csv` |
| **Person FP32 (Stock Clip, `person_walk`)** | Mac M1 | FP32 | 34.92 ms | 28.6 FPS | 39.34 ms | 1,008 MB | 196.5% (~2 cores)| PyTorch CPU runtime (imgsz 640, stock clip) | Logged in `benchmarks.csv` |
| **MOG2 (Stock Clip, `fire_smoke`)** | Mac M1 | - | 1.13 ms | 887.0 FPS | 1.64 ms | 715 MB | 97.8% (1 core) | Single CPU core (320 px downscaled, stock clip) | Logged in `benchmarks.csv` |
| **Fire/Smoke FP32 (Stock Clip, `fire_smoke`)** | Mac M1 | FP32 | 33.61 ms | 29.8 FPS | 40.64 ms | 1,044 MB | 194.9% (~2 cores)| PyTorch CPU runtime (imgsz 640, stock clip) | Logged in `benchmarks.csv` |

### INT8 Calibration Provenance & Accuracy Status:
* **Calibration Dataset Used**: The INT8 models were calibrated on Qualcomm AI Hub using **10 frames extracted from `test.mp4`** (a synthetic video generated by `scripts/make_test_video.py`).
* **Hardware Result**: Confirms that Qualcomm QNN successfully quantized and mapped 100% of operators (278/278 ops on YOLOv8n; 275/275 ops on Fire/Smoke) to the Hexagon NPU, delivering verified latency (8.13 ms / 6.47 ms) and memory numbers (13.6–16.6 MB).
* **Accuracy Caveat**: Because calibration data contained synthetic patterns rather than domain-specific human or smoke imagery, **INT8 detection accuracy (mAP drop) is unverified**. Re-quantizing on curated subsets of COCO validation and D-Fire test splits is required for production deployment.

---

## 3. Gated Pipeline Architecture & Gating Benchmarks

Continuous deep learning inference on every frame across multiple video streams is computationally prohibitive. Our architecture couples lightweight OpenCV background subtraction with rate-limited deep learning:

```
[ Video Frame Input (1280x720) ]
               │
               ▼
   [ Downscale to 320 px ] ────────► [ OpenCV MOG2 Motion Detector (~1.14 ms) ]
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        │                                                     │
                  [ Motion Detected ]                                   [ No Motion ]
                        │                                                     │
                        ▼                                                     ▼
        [ Run YOLOv8n Person Detector ]                             [ Skip Person Detection ]
             (Evaluated every N frames)                                 (Conserves NPU/CPU)
                        │
                        ▼
         [ Calculate Bounding Box Foot-Point ]
                        │
                        ▼
      [ cv2.pointPolygonTest(Zone, Foot) >= 0 ]
           ├── YES ──► Trigger INTRUSION Alert (Red Box [ZONE!])
           └── NO  ──► Safe Area (Green Box)

─────────────────────────────────────────────────────────────────────────────────────────────
[ Rate-Limited Fire & Smoke Detection (Decoupled Timer) ]
- Evaluated every M frames (e.g., every 10 frames = 3 times/sec at 30 FPS).
- Rationale: Fire and smoke diffuse slowly and lack sharp pixel displacement; decoupling from MOG2 prevents missed safety alarms.
```

### Full-Clip Gating Benchmarks (`videos/person_walk.mp4`):

The test clip `videos/person_walk.mp4` is a **60-frame video recorded at 2.0 FPS (total duration: 30.00 seconds)**.
* **Warmup Adaptation**: In a 30 FPS video, MOG2's default 30-frame warmup is 1.0s. At 2.0 FPS, 30 frames would consume 15.0s (50% of the clip). Pipeline was configured with `--warmup-seconds 2.0` (4–5 frames) to adapt across arbitrary frame rates.
* **Frame Accounting**:
  - Warmup frames (1–5): 5 frames (2.5s)
  - Motion frames (detected during walking): 11 frames (frames 8–13, 15–17, 20–21)
  - Static frames (person standing at gazebo): 44 frames (frames 6–7, 14, 18–19, 22–60)
  - $5 + 11 + 44 = 60 \text{ frames}$ accounted for.
* **Timing & Overhead Note**: Wall times on short 60-frame clips (1–3 seconds) are dominated by process startup, Python runtime, PyTorch model loading, and thread pool initialization. Accordingly, model invocation call counts provide the definitive hardware-neutral measurement for this clip.

| Pipeline Configuration | Key Flags | Person Model Calls | Detection & Intrusion Behavior |
|---|---|---|---|
| **1. Pure Gated** | `--person-interval 3 --hangover-seconds 0 --keepalive-seconds 0` | **4 calls** | Skips static frames (4 calls vs 20 baseline). **Vulnerability:** Misses intruder who stops moving before/at the zone (0 intrusion alerts). |
| **2. Production Gated** | `--person-interval 3 --hangover-seconds 3.0 --keepalive-seconds 1.0` | **34 calls** | **Catches stationary intruder:** Intrusion alert fires at **13.00s (frame 26)** and continues tracking intruder. |
| **3. Fair Ungated Baseline** | `--no-gate --person-interval 3` | **20 calls** | Evaluates fixed 1-in-3 frames regardless of motion ($60 / 3 = 20$). |
| **4. Continuous Baseline** | `--no-gate --person-interval 1` | **60 calls** | Evaluates every frame indiscriminately. |

> **Frame Rate Adaptation Note:** On this 2.0 FPS test clip, the 1.0s keepalive evaluates every 2 frames ($1.0 \times 2 = 2$). This is more frequent than interval 3 ($1.5\text{s}$), which is why production gating executes 34 calls on this specific low-FPS clip. On a standard **30 FPS surveillance camera**, keepalive is 1 call/s while interval 3 is 10 calls/s—achieving a **90% call reduction** during idle periods while guaranteeing stationary intruder detection.

---

## 4. Camera Capacity Model (~12 to 15 Cameras per Box)

Camera density on an edge box is constrained by two distinct subsystems: isolated NPU compute and host CPU video decoding.

### 1. NPU-Bound Budget (Inference Only):
* **Assumptions**: 30 FPS camera feeds.
  - Person model (INT8, 8.13 ms): evaluated every 3rd frame during active motion (10 calls/s) plus a 1-second keepalive during idle periods (1 call/s).
    $$\text{Person calls/s} = 10 \cdot f + 1 \cdot (1 - f)$$
    where $f$ is the active motion share (including hangover). Note: $f$ is an operational assumption, not a measurement.
  - Fire/smoke model (INT8, 6.47 ms): evaluated on a fixed interval of every 10 frames (3 calls/s) $\rightarrow 3 \times 6.47 \text{ ms} = \mathbf{19.41 \text{ ms/s}}$.
  - Safe duty cycle: **70% NPU utilization limit** (700 ms/s available compute budget) to prevent thermal throttling and queuing delay.

* **Scenario A: 25% Active Motion ($f = 0.25$)**:
  - Person calls/s: $10 \times 0.25 + 1 \times 0.75 = 2.5 + 0.75 = \mathbf{3.25 \text{ calls/s}}$
  - Person NPU time: $3.25 \times 8.13 \text{ ms} = \mathbf{26.42 \text{ ms/s}}$
  - Total NPU time/s per camera: $26.42 \text{ ms} + 19.41 \text{ ms} = \mathbf{45.83 \text{ ms/s}}$
  - Estimated NPU capacity at 70% load: $700 \text{ ms} / 45.83 \text{ ms} \approx \mathbf{15.27} \rightarrow \mathbf{\sim 15 \text{ cameras}}$.

* **Scenario B: 40% Active Motion ($f = 0.40$)**:
  - Person calls/s: $10 \times 0.40 + 1 \times 0.60 = 4.0 + 0.60 = \mathbf{4.60 \text{ calls/s}}$
  - Person NPU time: $4.60 \times 8.13 \text{ ms} = \mathbf{37.40 \text{ ms/s}}$
  - Total NPU time/s per camera: $37.40 \text{ ms} + 19.41 \text{ ms} = \mathbf{56.81 \text{ ms/s}}$
  - Estimated NPU capacity at 70% load: $700 \text{ ms} / 56.81 \text{ ms} \approx \mathbf{12.32} \rightarrow \mathbf{\sim 12 \text{ cameras}}$.

* **Estimated NPU capacity range**: **~12 to 15 cameras at 30 FPS** (inference-only, 70% NPU load).

### 2. CPU-Bound Budget (Decode, Pre/Post-processing, MOG2):
* On Mac M1, MOG2 (320 px) takes **1.14 ms/frame** on stock footage (~34.2 ms/s @ 30 FPS on 1 core).
* On the Edge Box (Kryo 670 CPU), decoding 12–15 concurrent H.264/H.265 RTSP streams in software would exhaust CPU resources.
* **Engineering Recommendation**: Multi-camera scaling requires hardware-accelerated video decoding (via GStreamer / Qualcomm multimedia accelerator `v4l2h264dec`) to ensure CPU decoding does not become the bottleneck before NPU saturation.

---

## 5. System Limitations & Known Constraints

To provide a fully transparent technical assessment, the following production limitations are acknowledged:

1. **Proof-of-Concept Evaluation Scale**: Validation was performed on individual stock test clips (`videos/person_walk.mp4` and `videos/fire_smoke.mp4`). While these successfully demonstrate functional correctness, comprehensive accuracy benchmarking across diverse lighting, weather, and camera angles requires dataset-scale evaluation.
2. **Alert De-Duplication**: Currently, alerts fire at each evaluation interval without stateful tracking. A commercial deployment would incorporate object tracking (e.g., ByteTrack) and temporal hysteresis to emit a single consolidated alarm per incident.
3. **Fire Model Precision & Recall**: The pre-trained D-Fire model (`rabahdev/fire-smoke-yolov8n`) has a published recall of ~0.688 and mAP50-95 of 0.362 on its test split. Missed detections or false alarms in complex lighting are expected.
4. **Licensing**: Ultralytics YOLOv8 models are licensed under AGPL-3.0. Commercial production deployments require purchasing an Ultralytics enterprise licence or migrating to an alternative model with a verified permissive licence (e.g., Apache-2.0 or BSD).
5. **Physical Hardware Access Pending**: Model inference was verified bare-metal on a hosted QCS6490 dev kit. Full system-level deployment (RTSP streaming, hardware V4L2 decoding, and thermal profiling) on the physical KavachX Edge Box remains pending hardware access.

---

## 6. How to Reproduce & Verify

All code, models, and outputs can be reproduced using standard commands:

```bash
# 1. Clone & setup environment
git clone https://github.com/Poi5eN/kavachx-edge.git
cd kavachx-edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download pre-trained weights
python scripts/download_models.py

# 3. Run intrusion detection on stock clip (with visual zone overlay)
python -m src.pipeline \
  --source videos/person_walk.mp4 \
  --zone "0.3,0.4;0.9,0.4;0.9,1.0;0.3,1.0" \
  --person-interval 3 \
  --person-conf 0.35 \
  --save outputs/person_walk_annotated.mp4

# 4. Run fire & smoke detection on stock clip
python -m src.pipeline \
  --source videos/fire_smoke.mp4 \
  --save outputs/fire_smoke_annotated.mp4

# 5. Execute standard benchmarks
python -m src.benchmark --mode motion --source videos/person_walk.mp4 --label mog2_test
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source videos/person_walk.mp4 --label person_test
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source videos/fire_smoke.mp4 --label fire_test
```
