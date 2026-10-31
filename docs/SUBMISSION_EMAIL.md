# KavachX Edge Detection Assignment — Submission Email Template

**To:** KavachX Hiring / Technical Team  
**From:** Gourav Kumar Upadhyay  
**Subject:** Submission: KavachX Edge Box AI Assignment (QCS6490) — Gourav Kumar Upadhyay  

---

Dear KavachX Team,

I have completed the Edge Box pre-trained model assignment for the Qualcomm QCS6490 chipset. 

All source code, bare-metal hardware benchmark data, architectural write-ups, and annotated video proofs are published in the GitHub repository below:

🔗 **GitHub Repository:** [https://github.com/Poi5eN/kavachx-edge](https://github.com/Poi5eN/kavachx-edge)  
📄 **Detailed Submission Report:** [https://github.com/Poi5eN/kavachx-edge/blob/main/docs/SUBMISSION_REPORT.md](https://github.com/Poi5eN/kavachx-edge/blob/main/docs/SUBMISSION_REPORT.md)

---

### Key Assignment Highlights & Results

1. **Pre-trained Models Deployed**:
   - **Fire & Smoke**: Fine-tuned YOLOv8n (`rabahdev/fire-smoke-yolov8n`). Class 0: `smoke`, Class 1: `fire`.
   - **Motion Detection**: OpenCV `BackgroundSubtractorMOG2` downscaled to 320 px (~1.1 ms latency).
   - **Intrusion Detection**: COCO YOLOv8n (Class 0: `person`) with foot-point tracking inside a user-configurable normalized polygon.

2. **Qualcomm QCS6490 Bare-Metal Benchmarks (Hexagon NPU)**:
   Inference was profiled directly on physical QCS6490 dev kit hardware (`Dragonwing RB3 Gen 2 Vision Kit`) using the Qualcomm AI Hub testbed:
   - **Person Detection (INT8)**: **8.13 ms** (123 FPS) — **10.36x speedup** vs FP32 (84.24 ms); peak RAM cut by 63% (16.6 MB). **100% of operators (278/278 ops) mapped to Hexagon NPU**.
   - **Fire & Smoke Detection (INT8)**: **6.47 ms** (154.5 FPS) — **10.02x speedup** vs FP32 (64.87 ms); peak RAM cut by 66% (13.6 MB). **100% of operators (275/275 ops) mapped to Hexagon NPU**.

3. **Optimized Gated Pipeline Architecture**:
   - Continuous YOLO inference burns excessive compute. Our pipeline runs lightweight MOG2 on every frame, gating the person detector to evaluate only during active motion (delivering a **2.2x to 3.3x speedup** over continuous inference).
   - The fire/smoke model is decoupled onto a fixed 10-frame timer, ensuring slow-billowing smoke is never missed even with zero pixel-level motion.

4. **Camera Capacity Analysis**:
   - Based on our isolated NPU inference budget (39.7 ms/s per 30 FPS camera) and reserving a safe 30% duty cycle margin for motion bursts and thermal headroom, the QCS6490 NPU can comfortably sustain **~13 to 17 concurrent 30 FPS camera streams**.
   - The report also outlines hardware video decoding requirements to ensure the host CPU does not bottleneck before NPU saturation.

5. **Video Proofs Included in Repo**:
   - `outputs/person_walk_annotated.mp4`: Demonstrates green bounding box outside the perimeter transitioning to a bright red `[ZONE!]` intrusion alert at 13.00s.
   - `outputs/fire_smoke_annotated.mp4`: Demonstrates smoke plume detection and bounding boxes.
   - `outputs/zone_preview.jpg`: Shows the virtual polygon overlay.

Full instructions to replicate all benchmarks and pipeline runs from scratch are detailed in the repository README.

I look forward to discussing the architecture, capacity trade-offs, and deployment details in our upcoming meeting!

Best regards,  
**Gourav Kumar Upadhyay**  
