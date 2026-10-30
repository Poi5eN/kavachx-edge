"""Generate a preview image showing the restricted zone polygon overlaid on a video frame."""
import argparse
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np


def parse_zone(text, w, h):
    if not text:
        return None
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";")]
    return np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)


def generate_zone_preview(source_path, zone_str, output_path="outputs/zone_preview.jpg"):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {source_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise SystemExit(f"Failed to read middle frame ({mid_frame_idx}) from {source_path}")

    h, w = frame.shape[:2]
    zone = parse_zone(zone_str, w, h)

    if zone is not None:
        # Draw semi-transparent filled polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 255))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Draw solid polygon boundary
        cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=3)

        # Draw label
        pt = zone[0]
        cv2.putText(frame, "RESTRICTED ZONE", (int(pt[0]), max(30, int(pt[1]) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Saved zone preview to: {output_path} (frame {mid_frame_idx}/{total_frames}, size: {w}x{h})")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Create restricted zone preview on video frame")
    ap.add_argument("--source", required=True, help="Path to video file")
    ap.add_argument("--zone", required=True, help='Normalised polygon coordinates "x,y;x,y;..."')
    ap.add_argument("--output", default="outputs/zone_preview.jpg", help="Path to save preview image")
    args = ap.parse_args()

    generate_zone_preview(args.source, args.zone, args.output)


if __name__ == "__main__":
    main()
