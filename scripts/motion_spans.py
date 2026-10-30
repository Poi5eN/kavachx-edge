"""Detect and print contiguous time spans of motion in a video using MotionDetector."""
import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.motion import MotionDetector


def find_motion_spans(video_path, min_area=800, warmup_frames=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0

    detector = MotionDetector(min_area=min_area, warmup_frames=warmup_frames)
    frame_idx = 0
    spans = []
    current_start = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        t = frame_idx / fps
        has_motion, fg, _ = detector.detect(frame)

        if has_motion:
            if current_start is None:
                current_start = t
        else:
            if current_start is not None:
                spans.append((current_start, t))
                current_start = None

    if current_start is not None:
        spans.append((current_start, frame_idx / fps))

    cap.release()

    print(f"\n========================================================")
    print(f"Video: {video_path}")
    print(f"Config: min_area={min_area}, warmup_frames={warmup_frames}")
    print(f"Total Frames: {frame_idx} | FPS: {fps:.2f} | Duration: {total_duration:.2f}s")
    print(f"--------------------------------------------------------")
    if not spans:
        print("No motion detected (after warmup).")
    else:
        print("Motion time spans (in seconds):")
        total_motion_sec = 0.0
        for i, (s, e) in enumerate(spans, 1):
            dur = e - s
            total_motion_sec += dur
            print(f"  Span {i}: {s:6.2f}s  -->  {e:6.2f}s  (duration: {dur:5.2f}s)")
        pct = (total_motion_sec / total_duration * 100) if total_duration > 0 else 0
        print(f"Total motion duration: {total_motion_sec:.2f}s ({pct:.1f}% of video)")
    print(f"========================================================")
    return spans


def main():
    ap = argparse.ArgumentParser(description="Find motion spans in a video")
    ap.add_argument("--source", required=True, help="Path to video file")
    ap.add_argument("--min-area", type=int, default=800, help="Minimum foreground pixel area (default 800)")
    ap.add_argument("--warmup-frames", type=int, default=30, help="Number of warmup frames to ignore (default 30)")
    args = ap.parse_args()
    find_motion_spans(args.source, min_area=args.min_area, warmup_frames=args.warmup_frames)


if __name__ == "__main__":
    main()
