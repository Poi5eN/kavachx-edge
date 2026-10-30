---
description: Run the standard benchmark set on one video and record results honestly
---
1. Ask which video to use and which machine this is running on (Mac M1, AI Hub QCS6490, or KavachX Edge Box).
2. Activate the environment: `source .venv/bin/activate`.
3. Run MOG2: `python -m src.benchmark --mode motion --source <video> --label mog2 --note "<machine>"`.
4. Run person model: `python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source <video> --label person_fp32 --note "<machine>"`.
5. Run fire/smoke model: `python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source <video> --label fire_fp32 --note "<machine>"`.
6. Run the pipeline gated and ungated and capture the printed summary lines.
7. Add every result to the table in `PROGRESS.md` with its source label. Never estimate or round up numbers.
8. Summarise what changed compared with the previous run in plain language.
