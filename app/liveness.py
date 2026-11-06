"""
Chống giả mạo thụ động (Passive Anti-Spoofing) bằng model MiniFASNet thật
(Silent-Face-Anti-Spoofing, minivision-ai, Apache-2.0 – bản ONNX từ yakhyo/face-anti-spoofing).

Ensemble 2 model, mỗi model nhìn khuôn mặt ở một tỉ lệ crop khác nhau:
  - MiniFASNetV2   : crop 2.7x bbox  -> 80x80
  - MiniFASNetV1SE : crop 4.0x bbox  -> 80x80
Đầu ra 3 lớp, lớp 1 = người thật (0 = ảnh in, 2 = phát lại qua màn hình).
Tiền xử lý giống bản gốc: ảnh BGR, giá trị 0-255 dạng float, KHÔNG chuẩn hoá.
"""
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import onnxruntime as ort

from .config import LIVENESS_MODEL_VERSION


class MiniFASNetDetector:
    def __init__(self, models_dir: Path):
        self.model_version = LIVENESS_MODEL_VERSION
        self.models = []
        for name, scale in (("MiniFASNetV2.onnx", 2.7), ("MiniFASNetV1SE.onnx", 4.0)):
            path = models_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Thiếu model anti-spoofing: {path}")
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            batch_dim = inp.shape[0]
            h, w = inp.shape[2], inp.shape[3]
            self.models.append({
                "session": sess,
                "input": inp.name,
                "output": sess.get_outputs()[0].name,
                "size": (int(w) if isinstance(w, int) else 80, int(h) if isinstance(h, int) else 80),
                "scale": scale,
                "dynamic_batch": not isinstance(batch_dim, int),
            })

    @staticmethod
    def _crop(image: np.ndarray, bbox: Sequence[float], scale: float, size: tuple) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        x, y, box_w, box_h = [float(v) for v in bbox]
        box_w = max(box_w, 1.0)
        box_h = max(box_h, 1.0)
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = x + box_w / 2, y + box_h / 2
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(src_w - 1, int(cx + new_w / 2))
        y2 = min(src_h - 1, int(cy + new_h / 2))
        crop = image[y1:y2 + 1, x1:x2 + 1]
        return cv2.resize(crop, size)

    def predict(self, image_bgr: np.ndarray, bbox_xywh: Sequence[float]) -> dict:
        """Trả về xác suất người thật (0..1) cho khuôn mặt tại bbox."""
        return self.predict_batch([(image_bgr, bbox_xywh)])[0]

    def predict_batch(self, items: Sequence[tuple]) -> list:
        """Chạy MiniFASNet theo batch cho danh sách (image_bgr, bbox_xywh) để tối ưu tốc độ CPU."""
        if not items:
            return []
        n = len(items)
        probs_sum = np.zeros((n, 3), dtype=np.float64)
        for m in self.models:
            crops = [
                np.transpose(self._crop(img, bbox, m["scale"], m["size"]).astype(np.float32), (2, 0, 1))
                for img, bbox in items
            ]
            if m["dynamic_batch"] or n == 1:
                batch_tensor = np.stack(crops, axis=0)
                logits = m["session"].run([m["output"]], {m["input"]: batch_tensor})[0]
            else:
                logits = np.vstack([
                    m["session"].run([m["output"]], {m["input"]: np.expand_dims(c, 0)})[0]
                    for c in crops
                ])
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            e = np.exp(logits_shifted)
            probs_sum += e / np.sum(e, axis=1, keepdims=True)
        probs_batch = probs_sum / len(self.models)
        results = []
        for i in range(n):
            probs = probs_batch[i]
            label = int(np.argmax(probs))
            results.append({
                "real_score": round(float(probs[1]), 4),
                "print_score": round(float(probs[0]), 4),
                "replay_score": round(float(probs[2]), 4),
                "label": "REAL" if label == 1 else ("PRINT" if label == 0 else "REPLAY"),
            })
        return results
