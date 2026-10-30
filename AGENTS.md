# AGENTS.md — KavachX Edge Detection Project

Shared instructions for every AI coding tool used on this repo (Antigravity, Claude Code, Cursor).
Read this file and `PROGRESS.md` at the start of every session. Update `PROGRESS.md` at the end.

## What this project is
A job assignment for KavachX: run three detections on their QCS6490 "Edge Box" using pre-trained models only.
1. Fire and smoke: YOLOv8n fine-tuned on D-Fire (`rabahdev/fire-smoke-yolov8n`, `best.pt`; class 0 = smoke, 1 = fire).
2. Motion: OpenCV `BackgroundSubtractorMOG2`.
3. Intrusion: COCO YOLOv8n, class 0 (person), foot point inside a restricted-zone polygon.
Also required: explore INT8 quantization, measure inference speed and CPU/NPU use, minimise resources per camera and maximise cameras per box.
Deadline for the build: end of day 2026-10-30. The owner then has a meeting where he will be questioned on everything, so keep code simple and explainable.

## Owner
Gourav Kumar Upadhyay (spell it Gourav). New to edge AI and computer vision: explain what you change and why, in plain language. Do not add complexity he cannot defend in an interview.

## Hard rules (do not break these)
1. **Never fabricate or estimate benchmark numbers.** Every number in a results table must come from a real run, with its source labelled: `Mac M1`, `AI Hub QCS6490`, or `KavachX Edge Box`.
2. **Mac numbers are not device numbers.** The assignment requires results from the QCS6490. Mac runs only prove the code works. Never present them as device results.
3. If something fails (install, export, NPU delegate), record the exact error and what was tried in `PROGRESS.md`. A documented failure is acceptable; hidden or invented success is not.
4. Do not commit model files, videos or `.venv` (see `.gitignore`). Keep the repo small.
5. Do not change the `results/benchmarks.csv` column names without noting it in `PROGRESS.md`, because older rows must stay comparable.
6. Ultralytics models are AGPL-3.0. Keep that limitation in the README.

## Layout
- `src/motion.py` MOG2 detector (320 px wide, 30-frame warmup, morphological open).
- `src/pipeline.py` gated pipeline: MOG2 every frame, person model during motion every N frames, fire model every M frames, zone intrusion check.
- `src/benchmark.py` one model or MOG2 on a video, appends a row to `results/benchmarks.csv`.
- `scripts/` download_models.py, make_test_video.py, export_int8.py, setup_github.sh.
- `models/` weights (git-ignored). `results/` CSVs and notes. `docs/` write-up material.
- `PROGRESS.md` shared tracker. `.agent/workflows/` saved Antigravity workflows.

## Environment (owner's Mac, as of 2026-10-30)
MacBook Pro M1, macOS, Python 3.14 in `.venv`, opencv-python 5.0.0, numpy 2.5.4, torch 2.15.0, ultralytics 8.4.193.
Always activate first: `source .venv/bin/activate`. Run modules from the repo root with `python -m src.<name>`.
The Edge Box (Qualcomm QCS6490, likely ARM Linux) is NOT available to the owner yet. Do not assume its OS, Python or SDK versions: ask or inspect.

## Common commands
```bash
python scripts/download_models.py
python scripts/make_test_video.py test.mp4
python -m src.benchmark --mode motion --source <video> --label <label> --note "<source>"
python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source <video> --label person_fp32 --note "<source>"
python -m src.benchmark --mode yolo --model models/fire-smoke-best.pt --source <video> --label fire_fp32 --note "<source>"
python -m src.pipeline --source <video> --zone "0.3,0.4;0.9,0.4;0.9,1.0;0.3,1.0" --show
python -m src.pipeline --source <video> --no-gate --person-interval 1     # baseline for comparison
```
Always use the same video across configurations so results are comparable. Record the `--note` source on every run.

## Definition of done for each phase
Phase is done only when: the command ran, the output or file exists, and the evidence is recorded in `PROGRESS.md`.

## Sync protocol between tools and people
- Start of session: `git pull`, read `AGENTS.md` and `PROGRESS.md`.
- During work: small commits with clear messages, one logical change each.
- End of session: update `PROGRESS.md` (phase statuses, session log, open questions, handoff notes), then commit and push.
- The owner also uses Claude in a chat window that cannot see this repo directly. He pastes `PROGRESS.md` there to re-sync, so keep it accurate, short and current.

## Style
Plain Python, standard library plus the packages in `requirements.txt`. Short functions, comments that explain why, no clever tricks. Type hints are optional.
