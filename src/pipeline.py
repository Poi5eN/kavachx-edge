"""Gated detection pipeline: cheap MOG2 every frame, YOLO only when needed.

 - Person (intrusion): YOLOv8n COCO, class 0, run only while motion is present.
 - Fire/smoke: fine-tuned YOLOv8n, run every --fire-interval frames even without
   motion (fire/smoke can be nearly static), so it is rate-limited instead of gated.
 - Intrusion: person's foot point (bottom-centre of box) inside --zone polygon.
"""
import argparse
import os
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


def draw_box(frame, box, label, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, y_text), (x1 + tw + 4, y_text + th + 4), color, -1)
    cv2.putText(frame, label, (x1 + 2, y_text + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video file, RTSP URL or camera index")
    ap.add_argument("--person-model", default="models/yolov8n.pt")
    ap.add_argument("--fire-model", default="models/fire-smoke-best.pt")
    ap.add_argument("--person-conf", type=float, default=0.4, help="confidence threshold for person detection")
    ap.add_argument("--fire-conf", type=float, default=0.3, help="confidence threshold for fire/smoke detection")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--person-interval", type=int, default=3, help="YOLO every N frames during motion")
    ap.add_argument("--fire-interval", type=int, default=10, help="fire model every N frames")
    ap.add_argument("--no-gate", action="store_true", help="disable motion gating (baseline)")
    ap.add_argument("--zone", default="", help='normalised polygon "x,y;x,y;..." e.g. "0.2,0.5;0.8,0.5;0.8,1;0.2,1"')
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="display video window (omit for headless)")
    ap.add_argument("--save", default="", help="path to save annotated video output (e.g. outputs/annotated.mp4)")
    ap.add_argument("--min-area", type=int, default=800, help="minimum foreground pixel count for motion (default 800)")
    ap.add_argument("--warmup-frames", type=int, default=30, help="warmup frames for background model (default 30)")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported late so motion.py works without it
    person = YOLO(args.person_model)
    fire = YOLO(args.fire_model)
    motion = MotionDetector(min_area=args.min_area, warmup_frames=args.warmup_frames)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    n = person_calls = fire_calls = motion_frames = 0
    t_person = t_fire = 0.0
    zone = None
    t0 = time.perf_counter()

    # Event tracking
    intrusion_count = 0
    first_intrusion_time = None
    detection_counts = {}
    first_detection_times = {}

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and n >= args.max_frames):
            break
        if zone is None and args.zone:
            zone = parse_zone(args.zone, frame.shape[1], frame.shape[0])
        n += 1
        video_time_s = n / fps
        has_motion, fg, _ = motion.detect(frame)
        motion_frames += has_motion

        person_boxes_to_draw = []
        if (has_motion or args.no_gate) and n % args.person_interval == 0:
            t = time.perf_counter()
            r = person(frame, classes=[0], conf=args.person_conf, imgsz=args.imgsz, verbose=False)[0]
            t_person += time.perf_counter() - t
            person_calls += 1
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [1.0] * len(boxes)
            for b, cf in zip(boxes, confs):
                alert = zone is not None and in_zone(b, zone)
                p_name = person.names.get(0, "person")
                if alert:
                    intrusion_count += 1
                    if first_intrusion_time is None:
                        first_intrusion_time = video_time_s
                    print(f"[frame {n} | {video_time_s:.2f}s] INTRUSION: person in restricted zone")
                if args.show or writer is not None:
                    color = (0, 0, 255) if alert else (0, 255, 0)
                    lbl = f"{p_name} {cf:.2f} [ZONE!]" if alert else f"{p_name} {cf:.2f}"
                    person_boxes_to_draw.append((b, lbl, color))

        fire_boxes_to_draw = []
        if n % args.fire_interval == 0:
            t = time.perf_counter()
            r = fire(frame, conf=args.fire_conf, imgsz=args.imgsz, verbose=False)[0]
            t_fire += time.perf_counter() - t
            fire_calls += 1
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else [1.0] * len(boxes)
            for b, c, cf in zip(boxes, clss, confs):
                cls_id = int(c)
                cls_name = fire.names.get(cls_id, f"class_{cls_id}")
                detection_counts[cls_name] = detection_counts.get(cls_name, 0) + 1
                if cls_name not in first_detection_times:
                    first_detection_times[cls_name] = video_time_s
                print(f"[frame {n} | {video_time_s:.2f}s] ALERT: {cls_name} detected ({cf:.2f})")
                if args.show or writer is not None:
                    color = (0, 0, 255) if cls_name == "fire" else (0, 165, 255)
                    fire_boxes_to_draw.append((b, f"{cls_name} {cf:.2f}", color))

        # Annotate frame after all inferences are done
        if args.show or writer is not None:
            if zone is not None:
                cv2.polylines(frame, [zone], True, (255, 0, 0), 2)
            for b, lbl, color in person_boxes_to_draw:
                draw_box(frame, b, lbl, color)
            for b, lbl, color in fire_boxes_to_draw:
                draw_box(frame, b, lbl, color)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("kavachx", frame)
            if cv2.waitKey(1) == 27:
                break

    if writer is not None:
        writer.release()
        print(f"saved annotated video to: {args.save}")

    if args.show:
        cv2.destroyAllWindows()

    wall = time.perf_counter() - t0
    print(f"\nframes={n} wall={wall:.1f}s overall_fps={n / max(wall, 0.001):.1f}")
    print(f"motion_frames={motion_frames} ({100 * motion_frames / max(n, 1):.0f}%)")
    print(f"person_calls={person_calls} avg={1000 * t_person / max(person_calls, 1):.1f}ms")
    print(f"fire_calls={fire_calls} avg={1000 * t_fire / max(fire_calls, 1):.1f}ms")

    # Event summaries
    print("\n--- Detection & Event Summary ---")
    if intrusion_count > 0:
        print(f"INTRUSIONS: {intrusion_count} events (first at {first_intrusion_time:.2f}s)")
    else:
        print("INTRUSIONS: 0 events")

    if detection_counts:
        for name, count in sorted(detection_counts.items()):
            first_t = first_detection_times.get(name, 0.0)
            print(f"{name.upper()}: {count} detections (first at {first_t:.2f}s)")
    else:
        print("FIRE/SMOKE: 0 detections")


if __name__ == "__main__":
    main()
