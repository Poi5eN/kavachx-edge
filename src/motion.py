"""Lightweight motion detection with OpenCV MOG2.

MOG2 models every pixel as a mixture of Gaussians. Pixels that do not fit the
learned background are marked foreground (255); shadows are 127 if enabled.
We run it on a downscaled frame because it is the cheapest way to cut CPU use.
"""
import cv2


class MotionDetector:
    def __init__(self, history=300, var_threshold=25, detect_shadows=False,
                 width=320, min_area=800, learning_rate=-1, warmup_frames=30):
        self.mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.width = width
        self.min_area = min_area          # in pixels of the downscaled frame
        self.learning_rate = learning_rate
        self.warmup_frames = warmup_frames  # background model is empty at start: ignore it
        self._seen = 0

    def detect(self, frame):
        """Return (motion: bool, foreground_pixels: int, mask)."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, int(h * self.width / w))),
                           interpolation=cv2.INTER_AREA)
        mask = self.mog.apply(small, learningRate=self.learning_rate)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)  # drop shadow (127)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)  # remove noise specks
        fg = cv2.countNonZero(mask)
        self._seen += 1
        if self._seen <= self.warmup_frames:
            return False, fg, mask
        return fg >= self.min_area, fg, mask
