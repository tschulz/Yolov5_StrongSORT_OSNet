import os

from threading import Thread

import sys
import math
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # yolov5 strongsort root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import cv2
import numpy as np
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import LOGGER

class LoadGstAppSink:
        def __init__(self, gst_pipeline='', img_size=640, stride=32, auto=True):
            self.mode = 'stream'
            self.img_size = img_size
            self.stride = stride
            self.gst_pipeline = gst_pipeline

            self.auto = auto
            # Start thread to read frames from video stream
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            assert self.cap.isOpened(), f'Failed to open {gst_pipeline}'
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
            self.frames = max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback

            _, self.img = self.cap.read()  # guarantee first frame

            self.thread = Thread(target=self.update, args=([]), daemon=True)
            LOGGER.info(f"Success ({self.frames} frames {w}x{h} at {self.fps:.2f} FPS)")
            self.thread.start()

        def update(self):
            # Read stream frames in daemon thread
            n, f, read = 0, self.frames, 1  # frame number, frame array, inference every 'read' frame
            while self.cap.isOpened() and n < f:
                n += 1
                # _, self.imgs[index] = self.cap.read()
                self.cap.grab()
                if n % read == 0:
                    success, im = self.cap.retrieve()
                    assert success, f'GStreamer appsink unresponsive'
                    self.img = im
                time.sleep(1 / self.fps)  # wait time

        def __iter__(self):
            self.count = -1
            return self

        def __next__(self):
            self.count += 1
            # if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):  # q to quit
            #     cv2.destroyAllWindows()
            #     raise StopIteration

            # Letterbox
            img0 = self.img.copy()
            img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)

            img = img[0].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW

            return self.gst_pipeline, img, img0, None, ''

        def __len__(self):
            return 1

