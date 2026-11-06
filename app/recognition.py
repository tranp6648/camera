"""
Phát hiện khuôn mặt (YuNet) + trích xuất embedding nhận diện (SFace) bằng OpenCV DNN.
Cả 2 model thuộc OpenCV Zoo (YuNet: MIT, SFace: Apache-2.0) – dùng thương mại được.
"""
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import FACE_DETECT_SCORE, MODEL_VERSION


class YuNetFaceDetector:
    def __init__(self, models_dir: Path):
        path = models_dir / "face_detection_yunet_2023mar.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu model phát hiện khuôn mặt: {path}")
        self.detector = cv2.FaceDetectorYN.create(str(path), "", (320, 320), FACE_DETECT_SCORE, 0.3, 50)

    def detect(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """Mỗi khuôn mặt: [x, y, w, h, 5 cặp landmark (10 số), score] – sắp xếp theo diện tích giảm dần."""
        h, w = image_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image_bgr)
        if faces is None:
            return []
        faces = sorted(list(faces), key=lambda f: float(f[2] * f[3]), reverse=True)
        return faces


class SFaceRecognizer:
    def __init__(self, models_dir: Path):
        path = models_dir / "face_recognition_sface_2021dec.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu model nhận diện khuôn mặt: {path}")
        self.model_version = MODEL_VERSION
        self.recognizer = cv2.FaceRecognizerSF.create(str(path), "")

    def extract_embedding(self, image_bgr: np.ndarray, face_row: np.ndarray) -> List[float]:
        aligned = self.recognizer.alignCrop(image_bgr, face_row)
        feat = self.recognizer.feature(aligned).flatten().astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            feat = feat / norm
        return [round(float(v), 6) for v in feat]

    @staticmethod
    def compute_similarity(vec1: Optional[List[float]], vec2: Optional[List[float]]) -> float:
        """Cosine similarity, trả về [-1, 1] làm tròn 4 chữ số (0.0 nếu vector không hợp lệ)."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        a = np.asarray(vec1, dtype=np.float32)
        b = np.asarray(vec2, dtype=np.float32)
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na == 0 or nb == 0:
            return 0.0
        return round(float(np.dot(a, b) / (na * nb)), 4)

    @staticmethod
    def average_embeddings(vectors: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        arr = np.asarray(vectors, dtype=np.float32)
        if weights and len(weights) == len(vectors) and sum(weights) > 0:
            w = np.asarray(weights, dtype=np.float32)
            mean = np.average(arr, axis=0, weights=w)
        else:
            mean = np.mean(arr, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        return [round(float(v), 6) for v in mean]
