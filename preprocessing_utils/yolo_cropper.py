from ultralytics import YOLO
import numpy as np

class YOLOCropper:
    def __init__(self, model_path, device="gpu"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def _load_if_needed(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def crop(self, pil_image, conf=0.25, imgsz=1028):
        self._load_if_needed()

        img_np = np.array(pil_image)

        results = self.model.predict(
            source=img_np,
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False,
            device=self.device,
        )[0]

        if len(results.boxes) == 0:
            return pil_image

        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0].tolist())
        return pil_image.crop((x1, y1, x2, y2))
