# KavachX Edge Detection System — Technical Submission Report

**Candidate:** Gourav Kumar Upadhyay  
**Date:** 2026-10-31  
**GitHub Repository:** [https://github.com/Poi5eN/kavachx-edge](https://github.com/Poi5eN/kavachx-edge)  
**Target Hardware:** Qualcomm QCS6490 ("Edge Box" / Dragonwing RB3 Gen 2 Vision Kit, Hexagon v68 NPU)  

---

## 1. Executive Summary & Compliance Verdict

All three required computer vision detections have been implemented and validated on real video clips. Deep learning inference has been benchmarked directly on Qualcomm QCS6490 hardware at the model level via Qualcomm AI Hub. Full end-to-end pipeline execution on the physical KavachX Edge Box and on-device hardware utilization monitoring remain pending hardware access.

| Assignment Requirement | Implementation / Methodology | Verification Status | Key Result / Finding |
|---|---|---|---|
| **1. Fire & Smoke Detection** | YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`). Class 0: `smoke`, Class 1: `fire`. Rate-limited every 10 frames. | **Met on real clip** | Smoke plume detected from 5.00s. INT8 inference takes **6.47 ms** on QCS6490 NPU. |
| **2. Motion Detection** | OpenCV `BackgroundSubtractorMOG2` downscaled to 320 px width with morphological opening. | **Met on real clip** | Runs in **1.14 ms** per frame (>870 FPS) on Mac M1 (1 CPU core). |
| **3. Intrusion Detection** | Pre-trained COCO YOLOv8n (class 0: `person`). Bounding box foot-point tracked inside a user-defined normalized polygon. | **Met on real clip** | Intrusion alert triggered at **13.00s** on real walking clip. INT8 inference takes **8.13 ms** on QCS6490 NPU. |
| **4. Direct Testing on QCS6490** | Compiled and profiled bare-metal inference jobs on Qualcomm AI Hub cloud testbed targeting the **Dragonwing RB3 Gen 2 Vision Kit (QCS6490 chipset)**. | **Partially Met** | Tested model inference across FP32 and INT8 precisions. Full pipeline with RTSP video decode on physical KavachX Edge Box is pending. |
| **5. Integrate OpenCV for Motion** | Architected motion-gated detection pipeline in `src/pipeline.py`. MOG2 acts as low-power gatekeeper. | **Met in pipeline** | On `person_walk.mp4`, motion gating cuts person model calls by **80%** (4 calls vs 20 calls) compared to the ungated baseline with equal frame skip. |
| **6. Explore INT8 Quantization** | Applied post-training INT8 quantization (PTQ) targeting Hexagon v68 NPU vector/tensor accelerators (HTP/HVX). | **Explored, validation pending** | **10.36x speedup** on Person; **10.02x speedup** on Fire. 100% of operators compiled to NPU. Dataset-scale mAP validation pending. |
| **7. Measure Inference Speed & CPU/NPU Utilization** | Benchmarked latency, P95, throughput, compute-unit mapping, and RAM. | **Speed met; utilization not met** | AI Hub provides execution timing, peak memory, and operator compute-unit mapping, but does not report percentage NPU/CPU utilization. |
| **8. Maximize Camera Capacity & Minimize Resources** | Formulated an isolated NPU vs CPU budget model with safety margins. | **Estimate only** | **~13 to 17 concurrent 30 FPS cameras per box** at a safe 70% NPU duty cycle (on-device CPU decoding unmeasured). |

---

## 2. Hardware Benchmark Proof (Qualcomm QCS6490 & Mac M1)

Inference was profiled directly on physical **Qualcomm QCS6490 hardware** via Qualcomm AI Hub testbeds (`Dragonwing RB3 Gen 2 Vision Kit`, Hexagon v68 NPU):

*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is isolated model inference peak memory (`estimated_inference_peak_memory`).*

| Model Configuration | Target Device | Precision | Mean Latency | Throughput | P95 Latency | Peak RAM* | CPU % (Mac)* | Compute Unit Operator Breakdown | AI Hub Job Reference |
|---|---|---|---|---|---|---|---|---|---|
| **Person Detection** (COCO YOLOv8n) | AI Hub QCS6490 | FP32 | 84.24 ms | 11.9 FPS | 85.92 ms | 45.3 MB | - | 224 GPU, 15 CPU, 13 NPU ops (GPU fallback) | Profile: [`jgjyw1jvg`](https://workbench.aihub.qualcomm.com/jobs/jgjyw1jvg/)<br>Compile: [`jgl817q8p`](https://workbench.aihub.qualcomm.com/jobs/jgl817q8p/) |
| **Person Detection** (COCO YOLOv8n) | **AI Hub QCS6490** | **INT8** | **8.13 ms** | **123.0 FPS** | **8.84 ms** | **16.6 MB** | - | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** | Profile: [`j5mkdordp`](https://workbench.aihub.qualcomm.com/jobs/j5mkdordp/)<br>Compile: [`jgl81ojjp`](https://workbench.aihub.qualcomm.com/jobs/jgl81ojjp/)<br>Quant: [`j57n7kqvp`](https://workbench.aihub.qualcomm.com/jobs/j57n7kqvp/) |
| **Fire/Smoke Detection** (`best.pt`) | AI Hub QCS6490 | FP32 | 64.87 ms | 15.4 FPS | 66.35 ms | 39.7 MB | - | 222 GPU, 15 CPU, 13 NPU ops (GPU fallback) | Profile: [`jgzy4w1op`](https://workbench.aihub.qualcomm.com/jobs/jgzy4w1op/)<br>Compile: [`jgd4z0yrg`](https://workbench.aihub.qualcomm.com/jobs/jgd4z0yrg/) |
| **Fire/Smoke Detection** (`best.pt`) | **AI Hub QCS6490** | **INT8** | **6.47 ms** | **154.5 FPS** | **7.17 ms** | **13.6 MB** | - | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** | Profile: [`jgnq7o2kg`](https://workbench.aihub.qualcomm.com/jobs/jgnq7o2kg/)<br>Compile: [`j56mdrk6p`](https://workbench.aihub.qualcomm.com/jobs/j56mdrk6p/)<br>Quant: [`jp449mz8p`](https://workbench.aihub.qualcomm.com/jobs/jp449mz8p/) |
| **MOG2 (Real Clip, `person_walk`)** | Mac M1 | - | 1.14 ms | 874.0 FPS | 1.48 ms | 714 MB | 99.0% (1 core) | Single CPU core (320 px downscaled, real clip) | Logged in `benchmarks.csv` |
| **Person FP32 (Real Clip, `person_walk`)** | Mac M1 | FP32 | 34.92 ms | 28.6 FPS | 39.34 ms | 1,008 MB | 196.5% (~2 cores) | PyTorch CPU runtime (imgsz 640, real clip) | Logged in `benchmarks.csv` |
| **MOG2 (Real Clip, `fire_smoke`)** | Mac M1 | - | 1.13 ms | 887.0 FPS | 1.64 ms | 715 MB | 97.8% (1 core) | Single CPU core (320 px downscaled, real clip) | Logged in `benchmarks.csv` |
| **Fire/Smoke FP32 (Real Clip, `fire_smoke`)** | Mac M1 | FP32 | 33.61 ms | 29.8 FPS | 40.64 ms | 1,044 MB | 194.9% (~2 cores) | PyTorch CPU runtime (imgsz 640, real clip) | Logged in `benchmarks.csv` |

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

### Measured Real-Clip Gating Benchmarks (`videos/person_walk.mp4`, 60 frames):

To measure the true impact of motion gating fairly, we evaluated the pipeline on the real walking clip under matching frame-skip intervals (`--person-interval 3`):

1. **Fair Baseline (Ungated, interval 3)**:
   - Command: `python -m src.pipeline --source videos/person_walk.mp4 --no-gate --person-interval 3`
   - Invocations: **20 person model calls** (34.8 FPS).
2. **Gated Pipeline (MOG2 + interval 3)**:
   - Command: `python -m src.pipeline --source videos/person_walk.mp4 --person-interval 3`
   - Invocations: **4 person model calls** (37.4 FPS).
   - **Gating Benefit**: Motion gating filtered out **80% of deep learning invocations** (4 calls vs 20 calls), reserving inference compute strictly for when the person moved.
3. **Continuous Inference Baseline (Ungated, per-frame, interval 1)**:
   - Command: `python -m src.pipeline --source videos/person_walk.mp4 --no-gate --person-interval 1`
   - Invocations: **60 person model calls** (14.0 FPS).
   - Compared to continuous per-frame inference, our gated architecture delivers a **2.67x throughput improvement** (37.4 FPS vs 14.0 FPS) and eliminates **93% of inference calls**.

---

## 4. Camera Capacity Model (~13 to 17 Cameras per Box)

Camera density on an edge box is constrained by two distinct subsystems: isolated NPU compute and host CPU video decoding.

### 1. NPU-Bound Budget (Inference Only):
* **Person Detection (INT8)**: 8.13 ms per call. Under MOG2 gating (~25% active motion, evaluated every 3rd frame $\approx$ 2.5 calls/s) $\rightarrow \mathbf{20.3 \text{ ms/s}}$.
* **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames ($3 \text{ calls/s}$) $\rightarrow \mathbf{19.4 \text{ ms/s}}$.
* **Total NPU inference budget per 30 FPS camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/second}}$.
* **NPU Capacity at Safe 70% Duty Cycle** (reserving 30% headroom for thermal throttling and motion bursts):
  $$\text{Capacity} = \frac{700 \text{ ms available}}{39.7 \text{ ms / camera}} \approx \mathbf{\sim 17 \text{ cameras}}.$$
* **Under Higher Motion Activity** (~40% motion activity $\rightarrow \mathbf{52.0 \text{ ms/s}}$ load):
  $$\text{Capacity} = \frac{700 \text{ ms available}}{52.0 \text{ ms / camera}} \approx \mathbf{\sim 13 \text{ cameras}}.$$
* **Estimated Capacity Range**: **~13 to 17 cameras (at 70% NPU load)**.

### 2. CPU-Bound Budget (Decode, Pre/Post-processing, MOG2):
* On Mac M1, MOG2 on 320 px frames takes **1.14 ms per frame** (approx. 34.2 ms/s per 30 FPS stream on 1 core).
* On the Edge Box (Kryo 670 CPU), decoding 13–17 concurrent H.264/H.265 RTSP streams in software would exhaust CPU resources.
* **Engineering Recommendation**: Multi-camera scaling requires hardware-accelerated video decoding (via GStreamer / Qualcomm multimedia accelerator `v4l2h264dec`) to ensure CPU decoding does not become the bottleneck before NPU saturation.

---

## 5. System Limitations & Known Constraints

To provide a fully transparent technical assessment, the following production limitations are acknowledged:

1. **Proof-of-Concept Evaluation Scale**: Validation was performed on individual test clips (`videos/person_walk.mp4` and `videos/fire_smoke.mp4`). While these successfully demonstrate functional correctness, comprehensive accuracy benchmarking across diverse lighting, weather, and camera angles requires dataset-scale evaluation.
2. **Alert De-Duplication**: Currently, alerts fire at each evaluation interval without stateful tracking. A commercial deployment would incorporate object tracking (e.g., ByteTrack) and temporal hysteresis to emit a single consolidated alarm per incident.
3. **Fire Model Precision & Recall**: The pre-trained D-Fire model (`rabahdev/fire-smoke-yolov8n`) has a published recall of ~0.688 and mAP50-95 of 0.362 on its test split. Missed detections or false alarms in complex lighting are expected.
4. **Licensing**: Ultralytics YOLOv8 models are licensed under AGPL-3.0. Commercial production deployments require enterprise licensing or migrating to an Apache-2.0 alternative (such as YOLOv6 or RT-DETR).
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

# 3. Run intrusion detection on real clip (with visual zone overlay)
python -m src.pipeline \
  --source videos/person_walk.mp4 \
  --zone "0.3,0.4;0.9,0.4;0.9,1.0;0.3,1.0" \
  --person-interval 3 \
  --person-conf 0.35 \
  --save outputs/person_walk_annotated.mp4

# 4. Run fire & smoke detection on real clip
python -m src.pipeline \
  --source videos/fire_smoke.mp4 \
  --save outputs/fire_smoke_annotated.mp4

# 5. Execute standard benchmarks
python -m src.benchmark --mode motion --source videos/person_walk.mp4 --label mog2_test
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source videos/person_walk.mp4 --label person_test
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source videos/fire_smoke.mp4 --label fire_test
```
