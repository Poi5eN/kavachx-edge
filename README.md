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

| Config | Source | mean ms | FPS | P95 ms | Peak RAM* | Compute Units / Notes |
|---|---|---|---|---|---|---|
| **MOG2 (Motion)** | Mac M1 | 0.59 | 1,693 | 0.64 | 239 MB | CPU (1 core, 320 px downscaled) |
| **Person FP32 (QNN)** | AI Hub QCS6490 | 84.24 | 11.9 | 85.92 | 45.3 MB | GPU / CPU fallback (224 GPU, 15 CPU, 13 NPU ops) |
| **Person INT8 (NPU)** | AI Hub QCS6490 | **8.13** | **123.0** | 8.84 | **16.6 MB** | **100% Hexagon NPU (278 NPU ops, 10.36x speedup)** |
| **Fire/smoke FP32 (QNN)**| AI Hub QCS6490 | 64.87 | 15.4 | 66.35 | 39.7 MB | GPU / CPU fallback (222 GPU, 15 CPU, 13 NPU ops) |
| **Fire/smoke INT8 (NPU)**| AI Hub QCS6490 | **6.47** | **154.5** | 7.17 | **13.6 MB** | **100% Hexagon NPU (275 NPU ops, 10.02x speedup)** |
| **Full pipeline, gated** | Mac M1 | - | **81.7** | - | - | CPU (40% motion, 33 person runs, 25 fire runs) |
| **Full pipeline, no gate**| Mac M1 | - | 24.5 | - | - | CPU (250 person runs, 3.3x slower baseline) |

### Camera Capacity Analysis (Estimate, Not a Result)
*Results from Qualcomm AI Hub measure isolated model inference on a Dragonwing RB3 Gen 2 dev kit, not end-to-end box performance. On-device RTSP video decoding (H.264/H.265), frame resizing/color conversion, NMS post-processing, and OpenCV MOG2 on the QCS6490 CPU are currently unmeasured.*

1. **NPU-Bound Budget (Inference Only)**:
   - **Person Detection (INT8)**: 8.13 ms per call. Evaluated every 3rd frame during motion (~25% active motion $\approx$ 2.5 inf/s) $\rightarrow \mathbf{20.3 \text{ ms/s}}$.
   - **Fire/Smoke Detection (INT8)**: 6.47 ms per call. Evaluated every 10 frames (~3 inf/s) $\rightarrow \mathbf{19.4 \text{ ms/s}}$.
   - **Total NPU inference budget per 30 FPS camera**: $20.3 + 19.4 = \mathbf{\sim 39.7 \text{ ms/s}}$.
   - **Estimated NPU capacity at safe 70% NPU duty cycle**: $700 \text{ ms} / 39.7 \text{ ms} \approx \mathbf{\sim 17 \text{ cameras}}$. At higher motion activity (~35% motion $\rightarrow 52 \text{ ms/s}$ load), capacity is $700 / 52 \approx \mathbf{\sim 13 \text{ cameras}}$.
   - **Estimated capacity range**: **~13 to 17 cameras (at 70% NPU load)**.

2. **CPU-Bound Budget (Decode, Pre/Post-processing, MOG2)**:
   - On Mac M1, MOG2 (320 px) takes 0.59 ms/frame (17.7 ms/s @ 30 FPS).
   - On the Edge Box, handling 13–17 concurrent RTSP decode pipelines and OpenCV operations on the Kryo CPU may become the primary bottleneck before NPU saturation. End-to-end hardware testing on the physical KavachX Edge Box is needed to measure actual CPU and thermal headroom.

## Open Questions & Next Steps
1. **Real test clips**: Replace synthetic `test.mp4` with real camera footage (person walking in restricted zone, real fire/smoke).
2. **INT8 accuracy check**: Measure mAP drop between FP32 and INT8 on validation datasets (COCO val / D-Fire test).
3. **Calibration data provenance**: Curate a domain-specific calibration dataset rather than 10 test frames.
4. **CPU & NPU utilization on device**: Measure system-wide CPU/NPU load and video decode overhead.
5. **KavachX Edge Box run**: Deploy full pipeline on physical Edge Box hardware.

## Known limits
- Fire/smoke model: recall 0.688, fire mAP50-95 0.362 on D-Fire test split; expect misses.
- Models are AGPL-3.0 (Ultralytics); check licensing before commercial use.
- MOG2 is sensitive to lighting changes and camera shake; tune `var_threshold` and `min_area`.
