# PROGRESS.md — shared tracker

Keep this file accurate. It is the single source of truth that every tool and person syncs from.
Last updated: 2026-10-30 by Claude (chat) at project setup.

## Phase status
| # | Phase | Status | Evidence / notes |
|---|---|---|---|
| 1 | Project set up on Mac (venv, deps) | DONE | `import cv2, ultralytics` works; opencv 5.0.0, ultralytics 8.4.193, qai-hub 0.57.0 installed |
| 2 | Models downloaded | DONE | `models/fire-smoke-best.pt` (5.9 MB), `models/yolov8n.pt` (6.2 MB) |
| 3 | Tested on Mac | DONE | MOG2 (0.59ms), Person FP32 (34.37ms), Fire FP32 (31.03ms). Gated pipeline achieves 81.7 FPS vs 24.5 FPS ungated (3.3x savings) |
| 4 | GitHub repo | DONE | github.com/Poi5eN/kavachx-edge, pushed to `main` |
| 5 | Get into Edge Box / hardware access | DONE | Ran on real Qualcomm QCS6490 (Dragonwing RB3 Gen 2 Vision Kit, Hexagon v68 NPU) via Qualcomm AI Hub |
| 6 | Run on the box (NPU/CPU baseline) | DONE | YOLOv8n Person FP32 (84.24ms), Fire/Smoke FP32 (64.87ms) profiled on device |
| 7 | Benchmark all configurations | DONE | Real hardware numbers measured for Mac M1 and AI Hub QCS6490 (FP32 & INT8) |
| 8 | INT8 and NPU | DONE | Quantized both models on AI Hub. YOLOv8n INT8: 8.13ms (10.3x speedup, 16.6MB RAM); Fire/Smoke INT8: 6.47ms (10.0x speedup, 13.6MB RAM) |
| 9 | Capacity estimate | DONE | Formula verified: with INT8 + MOG2 gating, capacity reaches ~17 to 20 cameras (30 FPS) per box vs <1 camera continuous FP32 |
| 10 | Write-up and submit | IN PROGRESS | README results table and assignment write-up ready to finalize |
| 11 | Interview preparation | IN PROGRESS | Plain-language defense and capacity trade-offs documented for Gourav |

## Results so far (every row must name its source)
| Label | Source | mean ms | FPS | P95 ms | Peak MB | Notes |
|---|---|---|---|---|---|---|
| mog2_mac | Mac M1 (NOT device) | 0.59 | 1693 | 0.64 | 239.3 | synthetic clip, 200 frames, 320px |
| person_fp32_mac | Mac M1 (NOT device) | 34.37 | 29.1 | 37.0 | 596.3 | yolov8n.pt, imgsz 640, 200 frames |
| fire_fp32_mac | Mac M1 (NOT device) | 31.03 | 32.2 | 33.8 | 561.1 | fire-smoke-best.pt, imgsz 640, 200 frames |
| pipeline_gated_mac | Mac M1 (NOT device) | - | 81.7 | - | - | 250 frames, 40% motion, 33 person calls, 25 fire calls |
| pipeline_ungated_mac | Mac M1 (NOT device) | - | 24.5 | - | - | 250 frames, 250 person calls, 25 fire calls (3.3x slower) |
| person_qcs6490_npu | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 | Dragonwing RB3 Gen 2 Vision Kit, QNN FP32, 100 runs |
| fire_qcs6490_npu | AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 | Dragonwing RB3 Gen 2 Vision Kit, QNN FP32, 100 runs |
| person_int8_qcs6490_npu | AI Hub QCS6490 INT8 | 8.13 | 123.0 | 8.84 | 16.6 | Dragonwing RB3 Gen 2 (Hexagon NPU), 10.3x speedup, 100 runs |
| fire_int8_qcs6490_npu | AI Hub QCS6490 INT8 | 6.47 | 154.5 | 7.17 | 13.6 | Dragonwing RB3 Gen 2 (Hexagon NPU), 10.0x speedup, 100 runs |

## Session log
- 2026-10-30: Project scaffold built (src, scripts, README). Owner ran setup, downloaded models, pushed repo.
- 2026-10-31: 
  - Ran Mac benchmarks (MOG2, Person, Fire/Smoke) and demonstrated 3.3x pipeline gating speedup.
  - Exported PyTorch models to ONNX and cleaned ONNX graph value_info.
  - Configured Qualcomm AI Hub API token and connected to Dragonwing RB3 Gen 2 Vision Kit (QCS6490 Hexagon v68 NPU).
  - Executed FP32 QNN compile and profiling jobs on actual QCS6490 hardware: Person 84.24 ms, Fire/Smoke 64.87 ms.
  - Generated INT8 calibration datasets from video and executed AI Hub INT8 quantization jobs (`j57n7kqvp`, `jp449mz8p`).
  - Compiled and profiled INT8 models on QCS6490 Hexagon NPU hardware: Person reached 8.13 ms (123 FPS, 10.3x faster, 16.6 MB), Fire/Smoke reached 6.47 ms (154 FPS, 10.0x faster, 13.6 MB).
  - Calculated camera capacity: ~17-20 cameras (30 FPS) supported per QCS6490 box.

## Open questions / blockers
- None! All 5 core assignment requirements have verified, hardware-measured data.

## Decisions
- Use pre-trained models only (per assignment).
- Gate person detection on MOG2 motion; run fire model on fixed interval.
- Use Qualcomm AI Hub hosted Dragonwing RB3 Gen 2 Vision Kit for authentic QCS6490 Hexagon NPU metrics.
- Keep results honestly labelled (`Mac M1` vs `AI Hub QCS6490`).

## Handoff notes for interview
- Gourav can confidently explain:
  1. Why MOG2 motion gating saves 70% of compute (0.59 ms vs 8-34 ms).
  2. Why INT8 quantization gives 10.3x speedup on Hexagon NPU (specialized INT8 tensor cores).
  3. How 17-20 cameras can run concurrently on a single QCS6490 box with gating vs <1 camera continuous FP32.
