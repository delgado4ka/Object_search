import numpy as np
from PIL import Image
import torch  # Assuming YOLOv8 is based on PyTorch
from ultralytics import YOLO  # Import the YOLOv8 library


class ObjDetection:
    def __init__(self):
        self.model = self.load_model()

    @staticmethod
    def load_model():
        return YOLO("forensicsUI/models/YOLOv8L.pt")

    def detect(self, source, classes=None, tracker="bytetrack.yaml", persist=False, isImage=False):
        if classes is None:
            classes = list(self.model.names.keys())
        if isImage:
            return self.model(source, classes=classes, device="mps")
        return self.model.track(source, classes=classes, tracker=tracker, persist=persist, device="mps")
