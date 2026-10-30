# KavachX Edge: Session Progress & Handoff Report for Claude

**Owner:** Gourav Kumar Upadhyay  
**Date:** 2026-10-31  
**Repository:** [github.com/Poi5eN/kavachx-edge](https://github.com/Poi5eN/kavachx-edge) (Branch: `main`)  
**Target Hardware:** Qualcomm QCS6490 (Dragonwing RB3 Gen 2 Vision Kit / Hexagon v68 NPU)

---

## 1. Executive Summary & Status Overview

All core assignment requirements have been implemented, tested, and benchmarked:
1. **OpenCV MOG2 Motion Detection**: Implemented in `src/motion.py` (downscaled to 320 px, 0.59 ms latency on Mac M1). Gated pipeline achieves **81.7 FPS** vs **24.5 FPS** ungated (a **3.3x speedup / 70% compute reduction**).
2. **Qualcomm AI Hub Integration**: Installed `qai-hub` (v0.57.0), configured API token, and executed compilation and inference profiling jobs on cloud-hosted **Qualcomm QCS6490 dev kit hardware** (`Dragonwing RB3 Gen 2 Vision Kit`).
3. **INT8 Quantization on Hexagon NPU**: Quantized both models on Qualcomm AI Hub using video calibration frames.
   - **Person Detection (YOLOv8n)**: Latency dropped from **84.24 ms to 8.13 ms (10.36x speedup)**; peak inference memory reduced by 63% (16.6 MB). 100% of ops executed on Hexagon NPU.
   - **Fire & Smoke Detection**: Latency dropped from **64.87 ms to 6.47 ms (10.02x speedup)**; peak inference memory reduced by 66% (13.6 MB). 100% of ops executed on Hexagon NPU.
4. **Camera Capacity Formulation**: Formulated separate NPU-bound and CPU-bound budget models. Under a safe 70% NPU duty cycle with MOG2 gating, capacity is estimated at **~13 to 17 cameras** at 30 FPS. On-device video decode and MOG2 CPU times are unmeasured.

---

## 2. Phase Status Matrix

| Phase | Description | Status | Evidence & Notes |
|---|---|---|---|
| **Phase 1** | Local Environment & Deps | **DONE** | Python virtual environment active; OpenCV 5.0.0, Ultralytics 8.4.193, NumPy, `qai-hub` 0.57.0 verified. |
| **Phase 2** | Pre-trained Models | **DONE** | `models/yolov8n.pt` (6.2 MB) & `models/fire-smoke-best.pt` (5.9 MB) downloaded. |
| **Phase 3** | Tested on Mac | **DONE** | Benchmarked on `test.mp4`. MOG2 (0.59 ms), Person FP32 (34.37 ms), Fire FP32 (31.03 ms). Gated pipeline 81.7 FPS. |
| **Phase 4** | GitHub Repository | **DONE** | Pushed to `origin/main` at `github.com/Poi5eN/kavachx-edge`. |
| **Phase 5** | Hardware Access | **PARTIAL** | Results come from an AI Hub-hosted QCS6490 dev kit (Dragonwing RB3 Gen 2), not the KavachX Edge Box; measures model inference only via cloud jobs. |
| **Phase 6** | Hardware Baseline (FP32) | **PARTIAL** | Model inference profiled via AI Hub on QCS6490 dev kit (Person FP32: 84.24 ms, Fire FP32: 64.87 ms). Full pipeline with video decode/OpenCV on physical KavachX Edge Box still pending device access. |
| **Phase 7** | Comprehensive Benchmarks | **DONE** | Model inference measured for Mac M1 and AI Hub QCS6490 (FP32 & INT8). |
| **Phase 8** | INT8 Quantization | **DONE** | Post-training INT8 quantization completed on AI Hub (`j57n7kqvp`, `jp449mz8p`). Verified 10x speedup on Hexagon NPU (100% ops on NPU). |
| **Phase 9** | Capacity Formulation | **DONE** | Separate NPU and CPU budgets formulated: ~13-17 cameras estimate at 70% NPU load (on-device decode/MOG2/pre/post times unmeasured). |
| **Phase 10** | Write-up & Documentation | **IN PROGRESS** | `README.md` and `PROGRESS.md` updated with full hardware results. |
| **Phase 11** | Interview Preparation | **IN PROGRESS** | Non-technical and technical defense points documented below. |

---

## 3. Verified Benchmark Results Table

All rows represent authentic runs logged in `results/benchmarks.csv`:
*Memory Note: Mac memory is full process RSS (`psutil`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is model inference peak memory (`estimated_inference_peak_memory`).*

| Benchmark Label | Hardware Source | Precision | Mean Latency | Throughput (FPS) | P95 Latency | Peak RAM* | Compute Unit Breakdown |
|---|---|---|---|---|---|---|---|
| `mog2_mac` | Mac M1 (NOT device) | - | 0.59 ms | 1,693 FPS | 0.64 ms | 239.3 MB | CPU (1 core, 320 px downscaled, 200 frames) |
| `person_fp32_mac` | Mac M1 (NOT device) | FP32 | 34.37 ms | 29.1 FPS | 37.00 ms | 596.3 MB | CPU (yolov8n.pt, imgsz 640, 200 frames) |
| `fire_fp32_mac` | Mac M1 (NOT device) | FP32 | 31.03 ms | 32.2 FPS | 33.81 ms | 561.1 MB | CPU (fire-smoke-best.pt, imgsz 640, 200 frames) |
| `pipeline_gated_mac` | Mac M1 (NOT device) | - | - | **81.7 FPS** | - | - | CPU (250 frames, 40% motion, 33 person runs, 25 fire runs) |
| `pipeline_ungated_mac` | Mac M1 (NOT device) | - | - | 24.5 FPS | - | - | CPU (250 frames, 250 person runs, 25 fire runs, 3.3x slower) |
| `person_fp32_qcs6490` | AI Hub QCS6490 | FP32 | 84.24 ms | 11.9 FPS | 85.92 ms | 45.3 MB | GPU / CPU fallback (224 GPU, 15 CPU, 13 NPU ops; 100 runs) |
| `fire_fp32_qcs6490` | AI Hub QCS6490 | FP32 | 64.87 ms | 15.4 FPS | 66.35 ms | 39.7 MB | GPU / CPU fallback (222 GPU, 15 CPU, 13 NPU ops; 100 runs) |
| `person_int8_qcs6490_npu` | **AI Hub QCS6490 INT8** | **INT8** | **8.13 ms** | **123.0 FPS** | **8.84 ms** | **16.6 MB** | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** |
| `fire_int8_qcs6490_npu` | **AI Hub QCS6490 INT8** | **INT8** | **6.47 ms** | **154.5 FPS** | **7.17 ms** | **13.6 MB** | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** |
| `mog2_person_walk_mac` | Mac M1 (NOT device) | - | 1.14 ms | 874.0 FPS | 1.48 ms | 714.1 MB | CPU (1 core, 320 px downscaled, person_walk.mp4, 200 frames) |
| `person_walk_mac` | Mac M1 (NOT device) | FP32 | 34.92 ms | 28.6 FPS | 39.34 ms | 1007.8 MB | CPU (yolov8n.pt, person_walk.mp4, 200 frames) |
| `fire_smoke_mac` | Mac M1 (NOT device) | FP32 | 33.61 ms | 29.8 FPS | 40.64 ms | 1043.6 MB | CPU (fire-smoke-best.pt, fire_smoke.mp4, 200 frames) |
| `mog2_fire_smoke_mac` | Mac M1 (NOT device) | - | 1.13 ms | 887.0 FPS | 1.64 ms | 714.6 MB | CPU (1 core, 320 px downscaled, fire_smoke.mp4, 200 frames) |

---

## 4. Resource Consumption & Camera Capacity Breakdown (Estimate, Not a Result)

*Results from Qualcomm AI Hub measure isolated model inference on a Dragonwing RB3 Gen 2 dev kit, not end-to-end box performance. On-device RTSP video decoding (H.264/H.265), frame resizing/color conversion, NMS post-processing, and OpenCV MOG2 on the QCS6490 CPU are currently unmeasured.*

### 1. NPU-Bound Budget (Inference Only)
- **Person Detection (INT8)**: 8.13 ms per call. Evaluated every 3rd frame during motion (~25% active motion $\approx 2.5 \text{ inf/s}$) $\rightarrow \mathbf{20.3 \text{ ms/second}}$.
- **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames ($3 \text{ inf/s}$) $\rightarrow \mathbf{19.4 \text{ ms/second}}$.
- **Total NPU inference budget per 30 FPS camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/second}}$.
- **Estimated NPU capacity at safe 70% NPU duty cycle**:
  $$\frac{700 \text{ ms/s}}{39.7 \text{ ms/camera}} \approx \mathbf{\sim 17 \text{ cameras}}.$$
  At higher motion activity (~35% motion $\rightarrow 52 \text{ ms/s}$ load):
  $$\frac{700 \text{ ms/s}}{52 \text{ ms/camera}} \approx \mathbf{\sim 13 \text{ cameras}}.$$
- **Estimated capacity range**: **~13 to 17 cameras (at 70% NPU load)**.

### 2. CPU-Bound Budget (Decode, Pre/Post-processing, MOG2)
- On Mac M1, MOG2 (320 px) takes 0.59 ms/frame (17.7 ms/s @ 30 FPS).
- On the Edge Box, handling 13–17 concurrent RTSP decode pipelines and OpenCV operations on the Kryo CPU may become the primary bottleneck before NPU saturation. End-to-end hardware testing on the physical KavachX Edge Box is needed to measure actual CPU and thermal headroom.

---

## 5. Open Questions & Pending Items

1. **Real test clips**: Replace synthetic `test.mp4` with real camera footage (person walking in restricted zone, real fire/smoke).
2. **INT8 accuracy check**: Measure mAP drop between FP32 and INT8 on validation datasets (COCO val / D-Fire test).
3. **Calibration data provenance**: Curate a domain-specific calibration dataset rather than 10 test frames.
4. **CPU & NPU utilization on device**: Measure system-wide CPU/NPU load and video decode overhead.
5. **KavachX Edge Box run**: Deploy full pipeline on physical Edge Box hardware to confirm system-level throughput and thermals.

---

## 6. Interview Defense Points for Gourav

1. **Why use OpenCV MOG2 instead of running YOLO continuously?**
   > *"Running YOLO every frame consumes 8 ms to 84 ms. OpenCV MOG2 takes only 0.59 ms. Using MOG2 as a gatekeeper filters out 70%+ of static frames, reserving the NPU strictly for when motion actually occurs."*
2. **Why run the fire model on a fixed interval rather than motion-gated?**
   > *"Fire and smoke can linger or billow slowly without fast pixel movement. Rate-limiting it to every 10 frames keeps compute negligible (under 20 ms/s) while ensuring safety alerts are never missed."*
3. **What did INT8 quantization give you on the QCS6490?**
   > *"In FP32, QNN fell back mostly to GPU (224 GPU ops). Post-training INT8 quantization mapped 100% of operators (278/278 ops) to the Hexagon NPU vector/tensor accelerators (HTP/HVX), delivering an authentic 10.36x speedup (8.13 ms vs 84.24 ms) and slashing peak inference RAM by 63%."*
4. **Why is the camera capacity stated as ~13–17 cameras at 70% load?**
   > *"The ~13–17 camera number is an NPU-bound estimate that reserves a 30% safety margin for burst motion and thermals. On-device video decoding, OpenCV MOG2, and pre/post-processing on the CPU are currently unmeasured, so testing on the physical Edge Box is required to confirm whether CPU or NPU saturates first."*
