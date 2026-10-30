"""Gated detection pipeline: cheap MOG2 every frame, YOLO only when needed.

 - Person (intrusion): YOLOv8n COCO, class 0, run only while motion is present.
 - Fire/smoke: fine-tuned YOLOv8n, run every --fire-interval frames even without
   motion (fire/smoke can be nearly static), so it is rate-limited instead of gated.
 - Intrusion: person's foot point (bottom-centre of box) inside --zone polygon.
"""
import argparse
import time

import cv2
import numpy as np

from src.motion import MotionDetector


def parse_zone(text, w, h):
    if not text:
        return None
    pts = [tuple(float(v) for v in p.split(",")) for p in text.split(";")]
    return np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)


def in_zone(box, zone):
    x1, y1, x2, y2 = box
    foot = (float((x1 + x2) / 2), float(y2))
    return cv2.pointPolygonTest(zone, foot, False) >= 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, RTSP URL or camera index")
    ap.add_argument("--person-model", default="models/yolov8n.pt")
    ap.add_argument("--fire-model", default="models/fire-smoke-best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--person-interval", type=int, default=3, help="YOLO every N frames during motion")
    ap.add_argument("--fire-interval", type=int, default=10, help="fire model every N frames")
    ap.add_argument("--no-gate", action="store_true", help="disable motion gating (baseline)")
    ap.add_argument("--zone", default="", help='normalised polygon "x,y;x,y;..." e.g. "0.2,0.5;0.8,0.5;0.8,1;0.2,1"')
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported late so motion.py works without it
    person = YOLO(args.person_model)
    fire = YOLO(args.fire_model)
    motion = MotionDetector()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    n = person_calls = fire_calls = motion_frames = 0
    t_person = t_fire = 0.0
    zone = None
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        if zone is None and args.zone:
            zone = parse_zone(args.zone, frame.shape[1], frame.shape[0])
        n += 1
        has_motion, fg, _ = motion.detect(frame)
        motion_frames += has_motion

        if (has_motion or args.no_gate) and n % args.person_interval == 0:
            t = time.perf_counter()
            r = person(frame, classes=[0], conf=0.4, imgsz=args.imgsz, verbose=False)[0]
            t_person += time.perf_counter() - t
            person_calls += 1
            for b in r.boxes.xyxy.cpu().numpy():
                alert = zone is not None and in_zone(b, zone)
                if alert:
                    print(f"[frame {n}] INTRUSION: person in restricted zone")
                if args.show:
                    cv2.rectangle(frame, tuple(map(int, b[:2])), tuple(map(int, b[2:])),
                                  (0, 0, 255) if alert else (0, 255, 0), 2)

        if n % args.fire_interval == 0:
            t = time.perf_counter()
            r = fire(frame, conf=0.3, imgsz=args.imgsz, verbose=False)[0]
            t_fire += time.perf_counter() - t
            fire_calls += 1
            for c in r.boxes.cls.cpu().numpy():
                print(f"[frame {n}] ALERT: {'smoke' if int(c) == 0 else 'fire'} detected")

        if args.show:
            if zone is not None:
                cv2.polylines(frame, [zone], True, (255, 0, 0), 2)
            cv2.imshow("kavachx", frame)
            if cv2.waitKey(1) == 27:
                break

    wall = time.perf_counter() - t0
    print(f"\nframes={n} wall={wall:.1f}s overall_fps={n / wall:.1f}")
    print(f"motion_frames={motion_frames} ({100 * motion_frames / max(n, 1):.0f}%)")
    print(f"person_calls={person_calls} avg={1000 * t_person / max(person_calls, 1):.1f}ms")
    print(f"fire_calls={fire_calls} avg={1000 * t_fire / max(fire_calls, 1):.1f}ms")


if __name__ == "__main__":
    main()
