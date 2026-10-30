"""Benchmark one model (or MOG2 alone) on a video and append a row to results/benchmarks.csv.

Examples
  python -m src.benchmark --mode motion --source test.mp4 --label mog2_cpu
  python -m src.benchmark --mode yolo --model models/yolov8n.pt --classes 0 --source test.mp4 --label person_fp32_cpu
  python -m src.benchmark --mode yolo --model models/yolov8n_int8.tflite --source test.mp4 --label person_int8
"""
import argparse
import csv
import os
import platform
import time

import cv2
import numpy as np
import psutil

from src.motion import MotionDetector


def run(args):
    cap = cv2.VideoCapture(args.source)
    frames = []
    while len(frames) < args.frames + args.warmup:
        ok, f = cap.read()
        if not ok:
            if not frames:
                raise SystemExit("could not read any frames from source")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop short clips
            continue
        frames.append(f)

    if args.mode == "motion":
        det = MotionDetector()
        fn = lambda f: det.detect(f)
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        classes = [int(c) for c in args.classes.split(",")] if args.classes else None
        fn = lambda f: model(f, classes=classes, imgsz=args.imgsz, conf=0.25, verbose=False)

    for f in frames[:args.warmup]:
        fn(f)                                     # warmup is excluded from stats

    proc = psutil.Process()
    proc.cpu_percent(None)
    psutil.cpu_percent(None, percpu=True)
    lat = []
    t_all = time.perf_counter()
    for f in frames[args.warmup:]:
        t = time.perf_counter()
        fn(f)
        lat.append((time.perf_counter() - t) * 1000)
    wall = time.perf_counter() - t_all
    proc_cpu = proc.cpu_percent(None)             # % of one core, can exceed 100
    sys_cpu = float(np.mean(psutil.cpu_percent(None, percpu=True)))

    lat = np.array(lat)
    row = {
        "label": args.label, "mode": args.mode, "model": args.model if args.mode == "yolo" else "MOG2",
        "imgsz": args.imgsz if args.mode == "yolo" else 320, "frames": len(lat),
        "mean_ms": round(float(lat.mean()), 2), "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2), "fps": round(len(lat) / wall, 2),
        "proc_cpu_pct": round(proc_cpu, 1), "sys_cpu_pct_avg": round(sys_cpu, 1),
        "rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "device": f"{platform.machine()} {platform.system()}", "note": args.note,
    }
    os.makedirs("results", exist_ok=True)
    path = "results/benchmarks.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["motion", "yolo"], required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="models/yolov8n.pt")
    ap.add_argument("--classes", default="", help="comma list, e.g. 0 for person")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", default="")
    run(ap.parse_args())
