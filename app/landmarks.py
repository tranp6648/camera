"""
Phân tích cử động phía server bằng MediaPipe Tasks (Apache-2.0):
  - FaceLandmarker (478 điểm + blendshapes) -> EAR, eyeBlink score, góc quay đầu (yaw)
  - HandLandmarker (21 điểm/bàn tay)        -> số ngón tay đang giơ

Quy ước yaw (giống hệt client src/services/visionEngine.ts):
  ảnh gốc từ camera KHÔNG lật gương; yaw > 0 = người dùng quay sang PHẢI của họ,
  yaw < 0 = quay sang TRÁI của họ.
"""
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
NOSE_TIP = 1
FACE_EDGE_A = 234   # má bên phải của người (nằm bên trái ảnh không lật)
FACE_EDGE_B = 454   # má bên trái của người

Point = Tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(pts: List[Point], idx: Tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def estimate_yaw(pts: List[Point]) -> float:
    a, b, nose = pts[FACE_EDGE_A], pts[FACE_EDGE_B], pts[NOSE_TIP]
    width = b[0] - a[0]
    if abs(width) < 1e-6:
        return 0.0
    ratio = (nose[0] - a[0]) / width
    s = max(-1.0, min(1.0, (0.5 - ratio) * 2.0))
    return math.degrees(math.asin(s))


def yaw_from_matrix(matrix) -> Optional[float]:
    """Yaw từ ma trận biến đổi 4x4 của MediaPipe. Dấu được đảo để khớp quy ước (+ = phải của người dùng)."""
    try:
        m = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    except Exception:
        return None
    if not np.all(np.isfinite(m)):
        return None
    return -math.degrees(math.atan2(m[0, 2], m[2, 2]))


def _angle_deg(a: Point, b: Point, c: Point) -> float:
    """Góc ABC (độ) tại đỉnh B."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    norm = math.hypot(*ba) * math.hypot(*bc)
    if norm <= 1e-6:
        return 0.0
    cos_v = max(-1.0, min(1.0, (ba[0] * bc[0] + ba[1] * bc[1]) / norm))
    return math.degrees(math.acos(cos_v))


def count_fingers(pts: List[Point]) -> Tuple[int, int, bool]:
    """Trả về (tổng số ngón giơ, số ngón không tính cái, ngón cái có giơ không).
    Kết hợp tỉ lệ khoảng cách + góc khớp để tránh đếm nhầm ngón cái đang khép hờ khi giơ 2-3 ngón.
    Khi 4 ngón dài đã xoè (bàn tay mở) dùng ngưỡng ngón cái nới hơn — góc camera hay làm
    ngón cái trông cong/khép dù người dùng đang giơ đủ 5.
    """
    wrist = pts[0]
    palm = max(_dist(pts[0], pts[9]), 1e-6)
    extended = 0
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        straight = _angle_deg(pts[mcp], pts[pip], pts[tip]) >= 145.0
        if (
            _dist(wrist, pts[tip]) > _dist(wrist, pts[pip]) * 1.14
            and _dist(pts[tip], pts[mcp]) > 0.55 * palm
            and straight
        ):
            extended += 1

    tip, ip, mcp_t, cmc = pts[4], pts[3], pts[2], pts[1]
    pinky_mcp, index_mcp, mid_mcp = pts[17], pts[5], pts[9]

    if extended == 4:
        # Xoè bàn tay: tip ra ngoài hơn IP so với lòng bàn, không cần góc khớp quá thẳng
        tip_out = _dist(tip, mid_mcp) > _dist(ip, mid_mcp) * 1.02
        tip_vs_pinky = _dist(tip, pinky_mcp) > _dist(mcp_t, pinky_mcp) * 1.08
        tip_vs_index = _dist(tip, index_mcp) > 0.32 * palm
        tip_vs_wrist = _dist(wrist, tip) > _dist(wrist, ip) * 1.02
        thumb = tip_out and tip_vs_pinky and tip_vs_index and tip_vs_wrist
    else:
        thumb_straight = (
            _angle_deg(cmc, mcp_t, tip) >= 148.0 and _angle_deg(mcp_t, ip, tip) >= 145.0
        )
        thumb = (
            _dist(tip, pinky_mcp) > _dist(mcp_t, pinky_mcp) * 1.30
            and _dist(tip, index_mcp) > 0.52 * palm
            and _dist(tip, mid_mcp) > 0.58 * palm
            and thumb_straight
        )
    return extended + (1 if thumb else 0), extended, thumb


class LandmarkAnalyzer:
    def __init__(self, models_dir: Path):
        face_path = models_dir / "face_landmarker.task"
        hand_path = models_dir / "hand_landmarker.task"
        for p in (face_path, hand_path):
            if not p.exists():
                raise FileNotFoundError(f"Thiếu model MediaPipe: {p}")
        self.face = mp_vision.FaceLandmarker.create_from_options(
            mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=2,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
        )
        self.hand = mp_vision.HandLandmarker.create_from_options(
            mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(hand_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=2,
                # IMAGE mode không có tracking như client (VIDEO) -> hạ ngưỡng dò lòng bàn tay
                min_hand_detection_confidence=0.3,
                min_hand_presence_confidence=0.3,
            )
        )

    def analyze(
        self,
        image_rgb: np.ndarray,
        want_hands: bool,
        hand_multi_scale: bool = False,
        want_face: bool = True,
    ) -> dict:
        h, w = image_rgb.shape[:2]
        out: dict = {"mp_face_count": 0, "ear": None, "blink": None, "yaw": None,
                     "hand_count": 0, "fingers": None, "fingers_no_thumb": None, "thumb": None}

        if want_face:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image_rgb))
            face_res = self.face.detect(mp_image)
            faces = face_res.face_landmarks or []
            out["mp_face_count"] = len(faces)
            if faces:
                # Chọn khuôn mặt lớn nhất
                def face_width(lms) -> float:
                    xs = [lm.x for lm in lms]
                    return max(xs) - min(xs)
                best_i = max(range(len(faces)), key=lambda i: face_width(faces[i]))
                pts = [(lm.x * w, lm.y * h) for lm in faces[best_i]]
                ear = (eye_aspect_ratio(pts, LEFT_EYE) + eye_aspect_ratio(pts, RIGHT_EYE)) / 2.0
                out["ear"] = round(ear, 4)
                yaw = None
                mats = getattr(face_res, "facial_transformation_matrixes", None)
                if mats and len(mats) > best_i:
                    yaw = yaw_from_matrix(mats[best_i])
                out["yaw"] = round(yaw if yaw is not None else estimate_yaw(pts), 2)
                if face_res.face_blendshapes and len(face_res.face_blendshapes) > best_i:
                    scores = {c.category_name: c.score for c in face_res.face_blendshapes[best_i]}
                    if "eyeBlinkLeft" in scores and "eyeBlinkRight" in scores:
                        out["blink"] = round((scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2.0, 4)

        if want_hands:
            # Frame NEUTRAL chỉ cần dò nhanh 1 lần trên ảnh gốc; chỉ quét đa tỷ lệ trên frame ACTION
            hands = self._detect_hands(image_rgb) if hand_multi_scale else self._run_hand(image_rgb)
            out["hand_count"] = len(hands)
            if hands:
                # Lấy bàn tay có số ngón giơ nhiều nhất (người dùng có thể để tay kia trong khung)
                best: Optional[Tuple[int, int, bool]] = None
                for pts in hands:
                    res = count_fingers(pts)
                    if best is None or res[0] > best[0]:
                        best = res
                out["fingers"], out["fingers_no_thumb"], out["thumb"] = best
        return out

    def _run_hand(self, rgb: np.ndarray) -> List[List[Point]]:
        h, w = rgb.shape[:2]
        res = self.hand.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return [[(lm.x * w, lm.y * h) for lm in lms] for lms in (res.hand_landmarks or [])]

    def _detect_hands(self, image_rgb: np.ndarray) -> List[List[Point]]:
        """Dò tay trên toàn ảnh; nếu trượt thì thử phóng to toàn ảnh rồi đến các vùng cắt chính.
        Tối ưu số lượng vùng cắt để không gây trễ trên CPU và không cần chạy lại FaceLandmarker."""
        hands = self._run_hand(image_rgb)
        if hands:
            return hands
        h, w = image_rgb.shape[:2]
        # Thử phóng to toàn ảnh 1.4x trước (rất hiệu quả khi bàn tay hơi nhỏ)
        up = cv2.resize(image_rgb, (int(w * 1.4), int(h * 1.4)), interpolation=cv2.INTER_CUBIC)
        hands = self._run_hand(up)
        if hands:
            return hands
        crops = (
            (0, int(h * 0.2), w, h),             # 80% dưới (tay thường thấp hơn mặt)
            (0, 0, int(w * 0.6), h),             # nửa trái
            (int(w * 0.4), 0, w, h),             # nửa phải
        )
        for x0, y0, x1, y1 in crops:
            crop = image_rgb[y0:y1, x0:x1]
            scale = 640.0 / max(crop.shape[:2])
            if scale > 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            hands = self._run_hand(crop)
            if hands:
                return hands
        return []
