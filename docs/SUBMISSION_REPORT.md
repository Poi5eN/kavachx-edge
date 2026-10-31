# KavachX Edge Detection System — Project Submission Report

**Candidate:** Gourav Kumar Upadhyay  
**Date:** 2026-10-31  
**GitHub Repository:** [https://github.com/Poi5eN/kavachx-edge](https://github.com/Poi5eN/kavachx-edge)  
**Target Hardware:** Qualcomm QCS6490 ("Edge Box" / Dragonwing RB3 Gen 2 Vision Kit, Hexagon v68 NPU)  

---

## 1. Executive Summary & Compliance Matrix

This report documents the design, bare-metal hardware benchmarking, quantization, and edge pipeline implementation for the KavachX Edge Box assignment. All tasks have been implemented strictly using existing pre-trained models with zero training from scratch, adhering to an honest benchmarking protocol.

| Assignment Requirement | Implementation / Methodology | Hardware Verification Status | Result Summary |
|---|---|---|---|
| **1. Fire & Smoke Detection** | YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`). Class 0: `smoke`, Class 1: `fire`. | **Verified on QCS6490 NPU & Mac M1** | **6.47 ms** (154.5 FPS, INT8) on Hexagon NPU. Rate-limited at 10-frame intervals. |
| **2. Motion Detection** | OpenCV `BackgroundSubtractorMOG2` downscaled to 320 px width with morphological opening. | **Verified on Mac M1 & Real Video Clips** | **1.14 ms** per frame (>870 FPS). Filters out 70%+ of idle video frames. |
| **3. Intrusion Detection** | Pre-trained COCO YOLOv8n (class 0: `person`). Bounding box foot-point tracked against a user-defined normalized polygon. | **Verified on QCS6490 NPU & Real Video Clips** | **8.13 ms** (123.0 FPS, INT8) on Hexagon NPU. Evaluated only during active motion. |
| **4. Direct Testing on QCS6490** | Compiled & profiled bare-metal inference jobs on Qualcomm AI Hub hosted Dragonwing RB3 Gen 2 dev kit (QCS6490 chipset). | **Verified on Bare-Metal Dev Kit Hardware** | Tested across FP32 and INT8 precisions with exact operator breakdown. |
| **5. OpenCV Integration** | Motion-gated detection pipeline in `src/pipeline.py`. MOG2 acts as low-power gatekeeper. | **Verified on Real Footage (`videos/`)** | Gated pipeline delivers **2.2x to 3.3x system throughput speedup** over continuous inference. |
| **6. INT8 Quantization** | Post-training INT8 quantization with representative calibration data. | **Verified on Hexagon v68 NPU** | **10.36x speedup** on Person; **10.02x speedup** on Fire. 100% of operators compiled to NPU. |
| **7. Resource & Utilization** | Measured latency, P95, throughput, compute-unit mapping, and RAM across all models. | **Verified across Mac M1 & QCS6490** | Peak inference RAM reduced by **63% to 66%** under INT8. |
| **8. Max Camera Capacity** | Two-tiered capacity budget: isolated NPU compute budget vs CPU decode/MOG2 budget. | **Formulated with Safety Margins** | **~13 to 17 concurrent 30 FPS camera streams** at safe 70% NPU duty cycle. |

---

## 2. Hardware Benchmark Proof (Qualcomm QCS6490)

Isolated model inference was profiled directly on physical **Qualcomm QCS6490 hardware** via Qualcomm AI Hub bare-metal testbeds (`Dragonwing RB3 Gen 2 Vision Kit`, Hexagon v68 NPU):

*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is isolated model inference peak memory (`estimated_inference_peak_memory`).*

| Model Configuration | Target Device | Precision | Mean Latency | Throughput | P95 Latency | Peak Memory* | Compute Unit Operator Breakdown |
|---|---|---|---|---|---|---|---|
| **Person Detection** (COCO YOLOv8n) | AI Hub QCS6490 | FP32 | 84.24 ms | 11.9 FPS | 85.92 ms | 45.3 MB | 224 GPU, 15 CPU, 13 NPU ops (GPU fallback) |
| **Person Detection** (COCO YOLOv8n) | **AI Hub QCS6490** | **INT8** | **8.13 ms** | **123.0 FPS** | **8.84 ms** | **16.6 MB** | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** |
| **Fire/Smoke Detection** (`best.pt`) | AI Hub QCS6490 | FP32 | 64.87 ms | 15.4 FPS | 66.35 ms | 39.7 MB | 222 GPU, 15 CPU, 13 NPU ops (GPU fallback) |
| **Fire/Smoke Detection** (`best.pt`) | **AI Hub QCS6490** | **INT8** | **6.47 ms** | **154.5 FPS** | **7.17 ms** | **13.6 MB** | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** |
| **MOG2 Motion Detection** | Mac M1 | - | 1.14 ms | 874.0 FPS | 1.48 ms | 714 MB RSS | Single CPU core (320 px downscaled, real clip) |
| **Full Pipeline (Gated)** | Mac M1 | FP32 | - | **81.7 FPS** | - | - | CPU (3.3x faster than continuous baseline) |
| **Full Pipeline (Continuous/Ungated)**| Mac M1 | FP32 | - | 24.5 FPS | - | - | CPU (continuous inference baseline) |

### Key Quantization Insights:
1. **The FP32 Bottleneck**: In FP32 precision, the Qualcomm Neural Network (QNN) runtime could not map the models to the NPU, falling back mostly to the Adreno GPU (224 ops on GPU). This resulted in high latency (~84 ms) and power consumption.
2. **100% Hexagon NPU Acceleration**: Post-training INT8 quantization allowed the Qualcomm compiler to map **every single neural network layer (278/278 ops)** directly to the Hexagon Tensor Processor (HTP) and Hexagon Vector Extensions (HVX). This unlocked a true **10.36x speedup** (8.13 ms) and cut inference memory by **63%**.

---

## 3. Gated Pipeline Architecture & Optimization

Running continuous deep learning inference on every frame across multiple 30 FPS camera streams quickly saturates any edge processor. Our architecture couples lightweight classical computer vision with targeted deep learning:

```
[ Video Frame Input (1280x720) ]
               │
               ▼
   [ Downscale to 320 px ] ────────► [ OpenCV MOG2 Motion Detector (~1.1 ms) ]
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

### Real Video Validation Results:
1. **Intrusion Detection (`videos/person_walk.mp4`)**:
   - The person walks from 2.0s to 12.0s outside the perimeter (green bounding box).
   - At **13.00s (frame 26)**, the person's feet cross the virtual fence polygon ($x \ge 0.30$).
   - The system immediately triggers `[frame 26 | 13.00s] INTRUSION: person in restricted zone` and switches the bounding box to solid red with the `[ZONE!]` badge.
2. **Smoke Detection (`videos/fire_smoke.mp4`)**:
   - MOG2 correctly reported 0 motion frames because smoke diffused smoothly.
   - The decoupled 10-frame timer evaluated the scene, detecting smoke plumes at 5.00s, 10.00s, 15.00s, 20.00s, 25.00s, and 30.00s with 50%–70% confidence.

---

## 4. Camera Capacity Model (~13 to 17 Cameras per Box)

To maximize camera density without crashing the edge device, we formulated a two-tier budget separating **NPU inference** from **CPU pre/post-processing**:

### 1. Isolated NPU Inference Budget:
* **Person Detection (INT8)**: 8.13 ms per call. Under MOG2 gating (~25% active motion, evaluated every 3rd frame $\approx$ 2.5 calls/s) $\rightarrow \mathbf{20.3 \text{ ms/s}}$.
* **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames ($3 \text{ calls/s}$) $\rightarrow \mathbf{19.4 \text{ ms/s}}$.
* **Total NPU inference compute per camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/second}}$.
* **NPU Capacity at Safe 70% Duty Cycle** (leaving 30% headroom for thermal throttling and motion bursts):
  $$\text{Capacity} = \frac{700 \text{ ms available}}{39.7 \text{ ms / camera}} \approx \mathbf{\sim 17 \text{ cameras}}.$$
  Under heavy motion scenes (~35% motion activity $\rightarrow 52 \text{ ms/s}$ load):
  $$\text{Capacity} = \frac{700 \text{ ms available}}{52.0 \text{ ms / camera}} \approx \mathbf{\sim 13 \text{ cameras}}.$$

### 2. CPU Video Decoding & Overhead Considerations:
* On the QCS6490 (8-core Kryo 670 CPU), decoding 13–17 concurrent H.264/H.265 RTSP streams in software would exhaust CPU resources.
* **Engineering Recommendation**: Use hardware video decoding (via GStreamer / Qualcomm multimedia accelerator `v4l2h264dec`) and run MOG2 on downscaled 320 px frames to ensure the CPU does not become the bottleneck before NPU saturation.

---

## 5. Repository Structure & Deliverables

All code, benchmarks, documentation, and visual outputs are committed to the repository:

```
kavachx-edge/
├── README.md                  # Comprehensive setup, run commands & benchmark tables
├── PROGRESS.md                # Phase-by-phase tracker and audit trail
├── requirements.txt           # Verified package dependencies
├── src/
│   ├── motion.py              # OpenCV MOG2 detector (320 px downscaled, morphological open)
│   ├── pipeline.py            # Gated intrusion & rate-limited fire/smoke pipeline
│   └── benchmark.py           # Standardized latency, FPS, and RSS benchmark suite
├── scripts/
│   ├── download_models.py     # Automated weights downloader (D-Fire & COCO)
│   ├── qcs6490_aihub_benchmark.py # Qualcomm AI Hub automated cloud testbed runner
│   ├── export_int8.py         # Post-training quantization export utility
│   ├── motion_spans.py        # Motion interval analyzer
│   └── zone_preview.py        # Virtual polygon visualizer
├── outputs/                   # H.264 annotated video proofs & preview images
│   ├── zone_preview.jpg       # Visual overlay of restricted zone polygon
│   ├── person_walk_annotated.mp4 # Video proof: green box outside -> red [ZONE!] alert
│   └── fire_smoke_annotated.mp4   # Video proof: smoke plume detection & bounding boxes
├── results/
│   └── benchmarks.csv         # Raw, unmodified CSV of every benchmark run
└── docs/
    ├── SUBMISSION_REPORT.md   # This comprehensive report
    └── HANDOFF_CLAUDE.md      # Detailed session notes and interview defense guide
```

---

## 6. How to Reproduce & Verify

To run the pipeline and replicate results on any machine:

```bash
# 1. Clone & activate environment
git clone https://github.com/Poi5eN/kavachx-edge.git
cd kavachx-edge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download pre-trained model weights
python scripts/download_models.py

# 3. Run intrusion detection on real clip (with visual zone overlay)
python -m src.pipeline \
  --source videos/person_walk.mp4 \
  --zone "0.3,0.4;0.9,0.4;0.9,1.0;0.3,1.0" \
  --person-interval 1 \
  --person-conf 0.35 \
  --save outputs/person_walk_annotated.mp4

# 4. Run fire and smoke detection on real clip
python -m src.pipeline \
  --source videos/fire_smoke.mp4 \
  --save outputs/fire_smoke_annotated.mp4

# 5. Execute standard benchmarks
python -m src.benchmark --mode motion --source videos/person_walk.mp4 --label mog2_test
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source videos/person_walk.mp4 --label person_test
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source videos/fire_smoke.mp4 --label fire_test
```
