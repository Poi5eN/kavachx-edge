"""Synthetic test clip: static scene, a box moves through the middle. Lets you test MOG2 anywhere."""
import sys
import cv2
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "test.mp4"
W, H, FPS, N = 640, 360, 25, 250
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
rng = np.random.default_rng(0)
bg = (rng.integers(90, 110, (H, W, 3))).astype(np.uint8)
for i in range(N):
    f = np.clip(bg.astype(np.int16) + rng.integers(-3, 4, bg.shape), 0, 255).astype(np.uint8)  # sensor noise
    if 100 <= i < 200:                                                                         # moving object
        x = 40 + (i - 100) * 5
        cv2.rectangle(f, (x, 140), (x + 60, 300), (30, 30, 200), -1)
    vw.write(f)
vw.release()
print("wrote", out)
