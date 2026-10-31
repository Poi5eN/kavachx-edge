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
| 5 | Get into Edge Box / hardware access | PARTIAL | Results come from an AI Hub-hosted QCS6490 dev kit (Dragonwing RB3 Gen 2), not the KavachX Edge Box; measures model inference only via cloud jobs |
| 6 | Run on the box (baseline) | PARTIAL | Model inference profiled via AI Hub on QCS6490 dev kit (Person FP32: 84.24ms, Fire FP32: 64.87ms). Full pipeline with video decode/OpenCV on physical KavachX Edge Box still pending device access |
| 7 | Benchmark all configurations | DONE | Model inference measured for Mac M1 and AI Hub QCS6490 (FP32 & INT8) |
| 8 | INT8 and NPU | DONE | Quantized both models on AI Hub. YOLOv8n INT8: 8.13ms (10.3x speedup, 16.6MB RAM); Fire/Smoke INT8: 6.47ms (10.0x speedup, 13.6MB RAM); 100% ops executed on Hexagon NPU |
| 9 | Capacity estimate | DONE | Separate NPU and CPU budgets formulated: ~13-17 cameras estimate at 70% NPU load (on-device decode/MOG2/pre/post times unmeasured) |
| 10 | Write-up and submit | DONE | Comprehensive docs/SUBMISSION_REPORT.md and docs/SUBMISSION_EMAIL.md generated with full results |
| 11 | Interview preparation | DONE | Defense points, technical/plain-English trade-offs, and email template prepared for Gourav |

## Results so far (every row must name its source)
*Memory Note: Mac memory is full process RSS (`psutil.Process().memory_info().rss`, includes Python runtime, PyTorch, OpenCV). AI Hub memory is model inference peak memory (`estimated_inference_peak_memory`).*

| Label | Source | mean ms | FPS | P95 ms | Peak MB* | Compute Unit Breakdown |
|---|---|---|---|---|---|---|
| mog2_mac | Mac M1 (NOT device) | 0.59 | 1693 | 0.64 | 239.3 | CPU (1 core, 320 px downscaled, synthetic clip, 200 frames) |
| person_fp32_mac | Mac M1 (NOT device) | 34.37 | 29.1 | 37.0 | 596.3 | CPU (yolov8n.pt, imgsz 640, 200 frames) |
| fire_fp32_mac | Mac M1 (NOT device) | 31.03 | 32.2 | 33.8 | 561.1 | CPU (fire-smoke-best.pt, imgsz 640, 200 frames) |
| pipeline_gated_mac | Mac M1 (NOT device) | - | 81.7 | - | - | CPU (250 frames, 40% motion, 33 person runs, 25 fire runs) |
| pipeline_ungated_mac | Mac M1 (NOT device) | - | 24.5 | - | - | CPU (250 frames, 250 person runs, 25 fire runs, 3.3x slower) |
| person_fp32_qcs6490 | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 | GPU / CPU fallback (224 GPU, 15 CPU, 13 NPU ops; 100 runs) |
| fire_fp32_qcs6490 | AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 | GPU / CPU fallback (222 GPU, 15 CPU, 13 NPU ops; 100 runs) |
| person_int8_qcs6490_npu | AI Hub QCS6490 INT8 | 8.13 | 123.0 | 8.84 | 16.6 | 100% Hexagon NPU (278 NPU ops, 10.36x speedup, 100 runs) |
| fire_int8_qcs6490_npu | AI Hub QCS6490 INT8 | 6.47 | 154.5 | 7.17 | 13.6 | 100% Hexagon NPU (275 NPU ops, 10.02x speedup, 100 runs) |
| mog2_person_walk_mac | Mac M1 (NOT device) | 1.14 | 874.0 | 1.48 | 714.1 | CPU (1 core, 320 px downscaled, person_walk.mp4, 200 frames) |
| person_walk_mac | Mac M1 (NOT device) | 34.92 | 28.6 | 39.34 | 1007.8 | CPU (yolov8n.pt, person_walk.mp4, 200 frames) |
| fire_smoke_mac | Mac M1 (NOT device) | 33.61 | 29.8 | 40.64 | 1043.6 | CPU (fire-smoke-best.pt, fire_smoke.mp4, 200 frames) |
| mog2_fire_smoke_mac | Mac M1 (NOT device) | 1.13 | 887.0 | 1.64 | 714.6 | CPU (1 core, 320 px downscaled, fire_smoke.mp4, 200 frames) |

### Camera Capacity Analysis (Estimate, Not a Result)
*On-device RTSP video decoding, frame resizing/normalization, NMS post-processing, and OpenCV MOG2 on the QCS6490 CPU are unmeasured. The following represents an isolated NPU vs CPU budget model:*

1. **NPU-Bound Budget (Inference Only)**:
   - **Person Detection (INT8)**: 8.13 ms per call. Under MOG2 gating (~25% active motion, evaluated every 3rd frame = ~2.5 calls/s) $\rightarrow \mathbf{20.3 \text{ ms/s}}$.
   - **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames (~3 calls/s) $\rightarrow \mathbf{19.4 \text{ ms/s}}$.
   - **Total NPU inference budget per 30 FPS camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/s}}$.
   - **Estimated NPU capacity at safe 70% NPU load**: $700 \text{ ms} / 39.7 \text{ ms} \approx \mathbf{\sim 17 \text{ cameras}}$. At higher motion activity (~35% motion $\rightarrow 52 \text{ ms/s}$ load), capacity is $700 / 52 \approx \mathbf{\sim 13 \text{ cameras}}$.
   - **Estimated capacity range**: **~13 to 17 cameras (at 70% NPU load)**.

2. **CPU-Bound Budget (MOG2, Video Decoding, Pre/Post-processing)**:
   - On Mac M1, MOG2 (320 px) takes 0.59 ms/frame (17.7 ms/s @ 30 FPS).
   - On QCS6490, multi-stream H.264/H.265 RTSP decoding and OpenCV MOG2 execution times on the Kryo CPU are currently **unmeasured**. If CPU decoding becomes a bottleneck before NPU saturation, camera count will be CPU-limited. Physical device profiling is needed to confirm the true ceiling.

## Session log
- 2026-10-30: Project scaffold built (src, scripts, README). Owner ran setup, downloaded models, pushed repo.
- 2026-10-31: 
  - Ran Mac benchmarks (MOG2, Person, Fire/Smoke) and demonstrated 3.3x pipeline gating speedup.
  - Exported PyTorch models to ONNX and cleaned ONNX graph value_info.
  - Configured Qualcomm AI Hub API token and profiled on Dragonwing RB3 Gen 2 Vision Kit dev kit.
  - Inspected compute-unit breakdown: FP32 models fell back mostly to GPU (224 GPU ops), whereas INT8 models compiled 100% to Hexagon NPU (278 NPU ops).
  - Profiled INT8 models on QCS6490 Hexagon NPU: Person reached 8.13 ms (123 FPS, 10.36x faster, 16.6 MB peak inference RAM); Fire/Smoke reached 6.47 ms (154 FPS, 10.02x faster, 13.6 MB peak inference RAM).
  - Formulated two-part capacity estimate: ~13-17 cameras at 70% NPU load (noting on-device CPU decode/MOG2 times remain unmeasured).

## Open questions / pending items
1. **Real test clips**: **DONE** — Validated on `person_walk.mp4` and `fire_smoke.mp4`; annotated H.264 videos saved to `outputs/`.
2. **INT8 accuracy check**: Measure mAP drop between FP32 and INT8 on validation splits (COCO val / D-Fire test) to verify detection precision.
3. **Calibration data provenance**: Current INT8 calibration used 10 frames from test clip; production deployment requires a curated, representative domain calibration set.
4. **CPU/NPU utilization & decode overhead**: Measure on-device CPU utilization and hardware video decoding (H.264/H.265 RTSP) under multi-stream load.
5. **KavachX Edge Box run**: Full end-to-end pipeline execution on the physical KavachX Edge Box to confirm system-level throughput and thermals.

## Decisions
- Use pre-trained models only (per assignment).
- Gate person detection on MOG2 motion; run fire model on fixed interval.
- Use Qualcomm AI Hub hosted Dragonwing RB3 Gen 2 Vision Kit dev kit for isolated model inference benchmarks.
- Keep results honestly labelled (`Mac M1` vs `AI Hub QCS6490`).

## Handoff notes for interview
- Gourav can confidently explain:
  1. Why MOG2 motion gating saves 70% of compute (0.59 ms vs 8-34 ms).
  2. Why INT8 quantization gives 10.3x speedup on Hexagon NPU (compiled to specialized Hexagon vector/tensor accelerators HTP/HVX; FP32 fell back to GPU).
  3. Why ~13-17 cameras is an NPU-bound estimate at 70% load, and why physical box testing is required to measure CPU decode/MOG2 overhead.
