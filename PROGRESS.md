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
| 5 | Get into Edge Box / hardware access | IN PROGRESS | Using Qualcomm AI Hub (`aihub.qualcomm.com`) to target virtual QCS6490 hardware. Needs user API token |
| 6 | Run on the box (CPU baseline) | TODO | Run ONNX/TFLite models on QCS6490 target device via AI Hub |
| 7 | Benchmark all configurations | IN PROGRESS | Mac baseline complete. Awaiting AI Hub device runs for QCS6490 NPU/CPU |
| 8 | INT8 and NPU | IN PROGRESS | Models exported to ONNX (`models/yolov8n.onnx`, `models/fire-smoke-best.onnx`). Local LiteRT had torchao conflict on Python 3.14; compiling/quantizing via Qualcomm AI Hub is the standard clean route |
| 9 | Capacity estimate | IN PROGRESS | Baseline capacity model formulated: 1 camera @ 30 FPS ungated requires ~1000ms/s of YOLO compute (saturation); motion gating cuts YOLO invocations by ~70%, increasing stream capacity by 3-4x |
| 10 | Write-up and submit | TODO | README results table + one-page write-up |
| 11 | Interview preparation | TODO | See Claude Docs runbook and learning path |

## Results so far (every row must name its source)
| Label | Source | mean ms | FPS | CPU % | Notes |
|---|---|---|---|---|---|
| mog2_mac | Mac M1 (NOT device) | 0.59 | 1693 | 100.0 (one core) | synthetic clip, 200 frames |
| person_fp32_mac | Mac M1 (NOT device) | 34.37 | 29.1 | 201.9 | yolov8n.pt, imgsz 640, 200 frames |
| fire_fp32_mac | Mac M1 (NOT device) | 31.03 | 32.23 | 202.4 | fire-smoke-best.pt, imgsz 640, 200 frames |
| pipeline_gated_mac | Mac M1 (NOT device) | - | 81.7 | - | 250 frames, 40% motion, 33 person calls, 25 fire calls |
| pipeline_ungated_mac | Mac M1 (NOT device) | - | 24.5 | - | 250 frames, 250 person calls, 25 fire calls (3.3x slower) |

## Session log
- 2026-10-30: Project scaffold built (src, scripts, README). Owner ran setup, downloaded models, pushed repo.
- 2026-10-31: Fixed workflow directory placement in active repo. Ran complete Mac benchmarks for MOG2, Person YOLOv8n, Fire/Smoke YOLOv8n, and compared gated vs ungated pipeline (81.7 FPS vs 24.5 FPS). Exported both models to ONNX. Installed `qai-hub` client. Tested LiteRT INT8 export (failed locally with torchao duplicate registration on Python 3.14; documented in tracker). Ready for AI Hub API key to run on cloud QCS6490 device.

## Open questions / blockers
- Qualcomm AI Hub API key needed from Gourav to run jobs on hosted QCS6490 hardware.
- Real test clips from phone (optional improvement for realistic visuals).

## Decisions
- Use pre-trained models only (per assignment).
- Gate the person model on MOG2 motion; run fire model on a fixed interval because fire/smoke can be nearly static.
- Honest labelling of every result source (see AGENTS.md hard rules).
- Use Qualcomm AI Hub for QCS6490 NPU/INT8 benchmarking since physical Edge Box hardware is not yet in hand.

## Handoff notes for the next session
- Configure AI Hub: run `qai-hub configure --api_token <TOKEN>` once obtained from aihub.qualcomm.com.
- Submit `models/yolov8n.onnx` and `models/fire-smoke-best.onnx` to QCS6490 on AI Hub with INT8 quantization.
