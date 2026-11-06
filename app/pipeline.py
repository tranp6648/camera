"""
Pipeline xác minh chấm công – server KHÔNG tin bất kỳ kết quả nào do client tự khai.

Client gửi một chuỗi frame gốc (không vẽ HUD) chụp trong lúc làm thử thách:
    [NEUTRAL(step 0), ACTION(step 0), NEUTRAL(step 1), ACTION(step 1), ..., FINAL]
Server tự:
  1. Phát hiện khuôn mặt (YuNet) trên từng frame, kiểm tra số lượng & kích thước mặt.
  2. Chạy MiniFASNet (anti-spoofing) trên các frame nhìn thẳng (NEUTRAL + FINAL).
  3. Đo EAR / eyeBlink / yaw / số ngón tay bằng MediaPipe và xác minh đúng hành động,
     đúng thứ tự mà server đã sinh ngẫu nhiên.
  4. Trích xuất embedding SFace: so khớp FINAL với hồ sơ đã đăng ký và kiểm tra
     mọi frame cùng một người (chống tráo người giữa chừng).
"""
import hashlib
import queue
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    BLINK_CLOSED_SCORE, BLINK_EAR_RATIO, BLINK_OPEN_MAX_SCORE, ENROLL_MAX_ABS_YAW,
    FACE_AI_WORKERS, LIVENESS_MODEL_VERSION, MAX_FACE_BRIGHTNESS, MAX_FRAMES,
    MAX_IMAGE_SIDE, MAX_SESSION_SPAN_MS, MIN_ACTION_GAP_MS, MIN_FACE_BRIGHTNESS,
    MIN_FACE_SHARPNESS, MIN_FACE_SIZE_PX, MIN_FRONTAL_SIMILARITY,
    MIN_LIVENESS_PER_FRAME, MIN_LIVENESS_THRESHOLD, MIN_SESSION_CONSISTENCY,
    MIN_SIMILARITY_THRESHOLD, MODEL_VERSION, MODELS_DIR, NEUTRAL_MAX_ABS_YAW, TURN_MIN_ABS_YAW,
)
from .landmarks import LandmarkAnalyzer
from .liveness import MiniFASNetDetector
from .recognition import SFaceRecognizer, YuNetFaceDetector

VALID_ACTIONS = {"BLINK", "TURN_LEFT", "TURN_RIGHT", "FINGERS_1", "FINGERS_2", "FINGERS_3", "FINGERS_5"}

REASON_MESSAGES = {
    "BAD_FRAME_SEQUENCE": "Chuỗi ảnh gửi lên không đúng cấu trúc thử thách",
    "DUPLICATE_FRAMES": "Phát hiện ảnh trùng lặp (nghi ngờ ảnh tĩnh)",
    "INVALID_TIMESTAMPS": "Thời gian chụp các frame không hợp lệ",
    "IMAGE_DECODE_FAILED": "Không đọc được ảnh",
    "NO_FACE": "Không tìm thấy khuôn mặt rõ ràng trong khung hình",
    "FACE_TOO_SMALL": "Khuôn mặt quá nhỏ – hãy đưa mặt lại gần camera",
    "FACE_TOO_BLURRY": "Ảnh khuôn mặt bị mờ/rung – vui lòng giữ yên khi chụp",
    "BAD_LIGHTING": "Ánh sáng khuôn mặt quá tối hoặc bị chói sáng",
    "POSE_TOO_EXTREME": "Góc nghiêng khuôn mặt quá lớn khi đăng ký",
    "MULTIPLE_FACES": "Có nhiều hơn một khuôn mặt trong khung hình",
    "SPOOF_DETECTED": "Phát hiện ảnh in / màn hình / video phát lại (MiniFASNet)",
    "ACTION_NOT_VERIFIED": "Server không xác minh được cử động theo yêu cầu",
    "IDENTITY_SWITCH": "Khuôn mặt thay đổi giữa các bước thử thách",
    "PROFILE_MISSING": "Nhân viên chưa đăng ký khuôn mặt",
    "PROFILE_MODEL_MISMATCH": "Hồ sơ khuôn mặt tạo bởi model cũ – cần đăng ký lại",
    "FACE_MISMATCH": "Khuôn mặt không khớp hồ sơ nhân viên",
    "INCONSISTENT_SAMPLES": "Các ảnh mẫu không cùng một người",
}


@dataclass
class FrameMeta:
    tag: str
    step: int
    ts: int


@dataclass
class FrameAnalysis:
    index: int
    tag: str
    step: int
    face_count: int = 0
    face_size: int = 0
    det_score: float = 0.0
    bbox: Optional[List[float]] = None
    sharpness: Optional[float] = None
    brightness: Optional[float] = None
    embedding: Optional[List[float]] = None
    spoof: Optional[dict] = None
    ear: Optional[float] = None
    blink: Optional[float] = None
    yaw: Optional[float] = None
    hand_count: int = 0
    fingers: Optional[int] = None
    fingers_no_thumb: Optional[int] = None
    error: Optional[str] = None

    def public(self) -> dict:
        return {
            "index": self.index, "tag": self.tag, "step": self.step,
            "face_count": self.face_count, "face_size": self.face_size,
            "sharpness": self.sharpness, "brightness": self.brightness,
            "real_score": self.spoof["real_score"] if self.spoof else None,
            "spoof_label": self.spoof["label"] if self.spoof else None,
            "ear": self.ear, "blink": self.blink, "yaw": self.yaw,
            "hand_count": self.hand_count, "fingers": self.fingers,
            "error": self.error,
        }


@dataclass
class Reasons:
    items: List[Dict[str, str]] = field(default_factory=list)

    def add(self, code: str, extra: str = "") -> None:
        msg = REASON_MESSAGES.get(code, code)
        if extra:
            msg = f"{msg} ({extra})"
        if not any(r["code"] == code and r["message"] == msg for r in self.items):
            self.items.append({"code": code, "message": msg})

    def has(self, *codes: str) -> bool:
        return any(r["code"] in codes for r in self.items)


class _PipelineWorker:
    """Mỗi worker giữ bộ instance YuNet + SFace + MediaPipe độc lập để xử lý song song không cần Global Lock."""
    def __init__(self, models_dir):
        self.detector = YuNetFaceDetector(models_dir)
        self.recognizer = SFaceRecognizer(models_dir)
        self.landmarks = LandmarkAnalyzer(models_dir)


class FaceVerificationPipeline:
    def __init__(self):
        # ONNX Runtime session cho MiniFASNet hoàn toàn thread-safe -> dùng chung & hỗ trợ batch inference
        self.anti_spoof = MiniFASNetDetector(MODELS_DIR)
        self._workers: "queue.Queue[_PipelineWorker]" = queue.Queue()
        for _ in range(FACE_AI_WORKERS):
            self._workers.put(_PipelineWorker(MODELS_DIR))

    @contextmanager
    def _acquire_worker(self):
        worker = self._workers.get()
        try:
            yield worker
        finally:
            self._workers.put(worker)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def decode_image(data: bytes) -> np.ndarray:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 64 or img.shape[1] < 64:
            raise ValueError("IMAGE_DECODE_FAILED")
        h, w = img.shape[:2]
        scale = MAX_IMAGE_SIDE / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def compute_face_quality(bgr: np.ndarray, bbox: Sequence[float]) -> Tuple[float, float]:
        """Trả về (sharpness Laplacian variance trên crop 112x112, brightness trung bình 0-255)."""
        h, w = bgr.shape[:2]
        x, y, bw, bh = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + max(1, bw)), min(h, y + max(1, bh))
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, 0.0
        gray = cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY)
        sharpness = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)
        brightness = round(float(np.mean(gray)), 1)
        return sharpness, brightness

    def analyze_frame(
        self,
        worker: _PipelineWorker,
        bgr: np.ndarray,
        index: int,
        tag: str,
        step: int,
        want_hands: bool,
        hand_multi_scale: bool,
        want_embedding: bool,
    ) -> FrameAnalysis:
        fa = FrameAnalysis(index=index, tag=tag, step=step)
        faces = worker.detector.detect(bgr)
        if faces:
            main = faces[0]
            main_area = float(main[2] * main[3])
            # Chỉ đếm khuôn mặt "đáng kể" (>= 25% diện tích mặt chính) để bỏ qua mặt rất nhỏ phía xa
            fa.face_count = sum(1 for f in faces if float(f[2] * f[3]) >= 0.25 * main_area)
            fa.face_size = int(min(main[2], main[3]))
            fa.det_score = round(float(main[-1]), 3)
            fa.bbox = [float(v) for v in main[:4]]
            fa.sharpness, fa.brightness = self.compute_face_quality(bgr, fa.bbox)
            if fa.face_size >= MIN_FACE_SIZE_PX and want_embedding:
                fa.embedding = worker.recognizer.extract_embedding(bgr, main)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        lm = worker.landmarks.analyze(
            rgb,
            want_hands=want_hands,
            hand_multi_scale=hand_multi_scale,
            want_face=True,
        )
        fa.ear, fa.blink, fa.yaw = lm["ear"], lm["blink"], lm["yaw"]
        fa.hand_count, fa.fingers, fa.fingers_no_thumb = lm["hand_count"], lm["fingers"], lm["fingers_no_thumb"]
        return fa

    # --------------------------------------------------------------- challenge
    @staticmethod
    def _check_step(action: str, neutral: FrameAnalysis, act: FrameAnalysis) -> dict:
        res = {"action": action, "passed": False, "neutral_frame": neutral.index, "action_frame": act.index}

        if action in ("TURN_LEFT", "TURN_RIGHT"):
            yn, ya = neutral.yaw, act.yaw
            res.update({"neutral_yaw": yn, "action_yaw": ya})
            if yn is None or ya is None:
                res["note"] = "Không đo được góc quay"
                return res
            direction = -1.0 if action == "TURN_LEFT" else 1.0
            neutral_ok = abs(yn) <= NEUTRAL_MAX_ABS_YAW
            action_ok = direction * ya >= TURN_MIN_ABS_YAW and direction * (ya - yn) >= TURN_MIN_ABS_YAW - 2
            res["passed"] = neutral_ok and action_ok
            if not neutral_ok:
                res["note"] = "Frame bắt đầu không nhìn thẳng"
            elif not action_ok:
                res["note"] = "Góc quay đầu chưa đủ hoặc sai hướng"
            return res

        if action == "BLINK":
            en, ea, bn, ba = neutral.ear, act.ear, neutral.blink, act.blink
            res.update({"neutral_ear": en, "action_ear": ea, "neutral_blink": bn, "action_blink": ba})
            if en is None or ea is None:
                res["note"] = "Không đo được độ mở mắt"
                return res
            open_ok = bn is None or bn <= BLINK_OPEN_MAX_SCORE
            closed_by_bs = ba is not None and bn is not None and ba >= BLINK_CLOSED_SCORE and (ba - bn) >= 0.2
            closed_by_ear = en > 0 and ea <= en * BLINK_EAR_RATIO
            yaw_ok = neutral.yaw is None or abs(neutral.yaw) <= NEUTRAL_MAX_ABS_YAW + 6
            res["passed"] = open_ok and (closed_by_bs or closed_by_ear) and yaw_ok
            if not open_ok:
                res["note"] = "Frame bắt đầu mắt không mở"
            elif not (closed_by_bs or closed_by_ear):
                res["note"] = "Không thấy mắt nhắm trong frame hành động"
            return res

        if action.startswith("FINGERS_"):
            target = int(action.split("_")[1])
            fa, fna = act.fingers, act.fingers_no_thumb
            res.update({"target": target, "action_fingers": fa, "neutral_fingers": neutral.fingers})
            if act.hand_count == 0 or fa is None:
                res["note"] = "Không thấy bàn tay trong frame hành động"
                return res
            # Kết hợp đếm tổng ngón (đã tính góc khớp ngón cái) và dung sai hẹp cho trường hợp giơ 1-3 ngón
            action_ok = fa == target or (target in (1, 2, 3) and fna == target and fa == target + 1)
            neutral_matches = neutral.fingers is not None and (
                neutral.fingers == target or (target in (1, 2, 3) and neutral.fingers_no_thumb == target)
            )
            res["passed"] = action_ok and not neutral_matches
            if not action_ok:
                res["note"] = f"Đếm được {fa} ngón, yêu cầu {target}"
            elif neutral_matches:
                res["note"] = "Cử chỉ đã có sẵn trước khi thử thách bắt đầu"
            return res

        res["note"] = "Hành động không hợp lệ"
        return res

    # ------------------------------------------------------------------ verify
    def verify_session(self, images: List[bytes], metas: List[FrameMeta], expected_actions: List[str],
                       registered_embedding: Optional[List[float]], registered_model_version: Optional[str]) -> dict:
        reasons = Reasons()
        n = len(images)
        structure_ok = True

        # 1) Cấu trúc chuỗi frame
        expected_actions = [a for a in expected_actions if isinstance(a, str)]
        if (n != len(metas) or n < 2 or n > MAX_FRAMES or not expected_actions
                or any(a not in VALID_ACTIONS for a in expected_actions)):
            structure_ok = False
        tags = [m.tag for m in metas]
        if structure_ok and (tags.count("FINAL") != 1 or tags[-1] != "FINAL"):
            structure_ok = False

        step_pairs: List[tuple] = []
        if structure_ok:
            last_idx = -1
            for step, _ in enumerate(expected_actions):
                ni = next((i for i, m in enumerate(metas) if m.tag == "NEUTRAL" and m.step == step), None)
                ai = next((i for i, m in enumerate(metas) if m.tag == "ACTION" and m.step == step), None)
                if ni is None or ai is None or not (last_idx < ni < ai):
                    structure_ok = False
                    break
                step_pairs.append((ni, ai))
                last_idx = ai
        if not structure_ok:
            reasons.add("BAD_FRAME_SEQUENCE")

        # Timestamps (chống gửi các frame chụp liên thanh < 120ms hoặc phiên kéo dài bất thường)
        ts = [m.ts for m in metas]
        if n >= 2 and (any(b < a for a, b in zip(ts, ts[1:])) or ts[-1] - ts[0] > MAX_SESSION_SPAN_MS):
            reasons.add("INVALID_TIMESTAMPS")
        for ni, ai in step_pairs:
            if ts[ai] - ts[ni] < MIN_ACTION_GAP_MS:
                reasons.add("INVALID_TIMESTAMPS", f"bước {metas[ni].step + 1}")

        # Ảnh trùng byte-by-byte
        hashes = [hashlib.sha256(b).hexdigest() for b in images]
        if len(set(hashes)) != len(hashes):
            reasons.add("DUPLICATE_FRAMES")

        # 2) Giải mã ảnh ngoài worker lock để tăng tốc xử lý đồng thời
        decoded_bgrs: List[Optional[np.ndarray]] = []
        for i, data in enumerate(images):
            try:
                decoded_bgrs.append(self.decode_image(data))
            except ValueError:
                decoded_bgrs.append(None)
                reasons.add("IMAGE_DECODE_FAILED", f"frame {i}")

        # Phân tích từng frame bằng worker pool (không giữ Global Lock toàn cục)
        finger_steps = {i for i, a in enumerate(expected_actions) if a.startswith("FINGERS_")}
        analyses: List[FrameAnalysis] = []
        with self._acquire_worker() as worker:
            for i, (bgr, meta) in enumerate(zip(decoded_bgrs, metas)):
                if bgr is None:
                    analyses.append(FrameAnalysis(index=i, tag=meta.tag, step=meta.step, error="IMAGE_DECODE_FAILED"))
                    continue
                frontal = meta.tag in ("NEUTRAL", "FINAL")
                finger_at_step = (
                    meta.step in finger_steps
                    and meta.step < len(expected_actions)
                    and expected_actions[meta.step].startswith("FINGERS_")
                )
                want_hands = finger_at_step and meta.tag in ("NEUTRAL", "ACTION")
                # Chỉ quét multi-scale tìm tay ở frame ACTION (NEUTRAL chỉ cần quét nhanh 1 lần)
                hand_multi_scale = finger_at_step and meta.tag == "ACTION"
                # Chỉ trích xuất SFace embedding trên frame nhìn thẳng (NEUTRAL + FINAL), tiết kiệm 33% CPU
                analyses.append(self.analyze_frame(
                    worker, bgr, i, meta.tag, meta.step,
                    want_hands=want_hands,
                    hand_multi_scale=hand_multi_scale,
                    want_embedding=frontal,
                ))

        # Chạy MiniFASNet theo batch (lock-free) trên tất cả các frame nhìn thẳng (NEUTRAL + FINAL)
        spoof_targets: List[Tuple[int, np.ndarray, Sequence[float]]] = []
        for fa in analyses:
            if fa.tag in ("NEUTRAL", "FINAL") and not fa.error and fa.face_size >= MIN_FACE_SIZE_PX and fa.bbox:
                bgr = decoded_bgrs[fa.index]
                if bgr is not None:
                    spoof_targets.append((fa.index, bgr, fa.bbox))
        if spoof_targets:
            spoof_results = self.anti_spoof.predict_batch([(img, bbox) for _, img, bbox in spoof_targets])
            for (idx, _, _), sp in zip(spoof_targets, spoof_results):
                analyses[idx].spoof = sp

        # Kiểm tra số lượng & chất lượng khuôn mặt trên từng frame
        for fa in analyses:
            if fa.error:
                continue
            require_face = fa.tag in ("NEUTRAL", "FINAL") or (
                fa.tag == "ACTION" and fa.step not in finger_steps
            )
            if fa.face_count == 0:
                if require_face:
                    reasons.add("NO_FACE", f"frame {fa.index}")
            elif fa.face_size < MIN_FACE_SIZE_PX:
                if require_face:
                    reasons.add("FACE_TOO_SMALL", f"frame {fa.index}")
            # Không cho phép nhiều khuôn mặt ở bất kỳ frame nào trong phiên (chống người thứ 2 làm hộ cử chỉ)
            if fa.face_count > 1:
                reasons.add("MULTIPLE_FACES", f"frame {fa.index}")

        # 3) Anti-spoofing trên các frame nhìn thẳng
        frontal_scores = [fa.spoof["real_score"] for fa in analyses if fa.tag in ("NEUTRAL", "FINAL") and fa.spoof]
        frontal_total = sum(1 for fa in analyses if fa.tag in ("NEUTRAL", "FINAL"))
        liveness_score = round(float(np.mean(frontal_scores)), 4) if frontal_scores else 0.0
        min_real = min(frontal_scores) if frontal_scores else 0.0
        anti_spoof_passed = (
            len(frontal_scores) == frontal_total and frontal_total > 0
            and liveness_score >= MIN_LIVENESS_THRESHOLD and min_real >= MIN_LIVENESS_PER_FRAME
        )
        if not anti_spoof_passed and frontal_scores:
            reasons.add("SPOOF_DETECTED", f"điểm người thật {liveness_score:.2f}, thấp nhất {min_real:.2f}")

        # 4) Xác minh hành động (bắt buộc không có MULTIPLE_FACES hay lỗi chuỗi frame)
        step_results = []
        if structure_ok and len(analyses) == n:
            for step, (ni, ai) in enumerate(step_pairs):
                step_results.append(self._check_step(expected_actions[step], analyses[ni], analyses[ai]))
        challenge_passed = (
            structure_ok and len(step_results) == len(expected_actions)
            and all(r["passed"] for r in step_results)
            and not reasons.has(
                "INVALID_TIMESTAMPS", "DUPLICATE_FRAMES", "NO_FACE",
                "FACE_TOO_SMALL", "MULTIPLE_FACES", "IMAGE_DECODE_FAILED",
            )
        )
        if structure_ok and not all(r["passed"] for r in step_results):
            failed = [f"bước {i + 1} {r['action']}: {r.get('note', '')}" for i, r in enumerate(step_results) if not r["passed"]]
            reasons.add("ACTION_NOT_VERIFIED", "; ".join(failed))

        # 5) Nhận diện danh tính đa khung hình (Multi-Frame SFace Fusion: kết hợp NEUTRAL + FINAL)
        final = analyses[-1] if analyses else None
        final_emb = final.embedding if final else None
        identity_frames = [fa for fa in analyses if fa.tag in ("NEUTRAL", "FINAL") and not fa.error]
        frontal_embs = [fa.embedding for fa in identity_frames if fa.embedding and fa.face_count == 1]
        frontal_weights = [1.5 if fa.tag == "FINAL" else 1.0 for fa in identity_frames if fa.embedding and fa.face_count == 1]

        similarity = 0.0
        matched = False
        if not registered_embedding:
            reasons.add("PROFILE_MISSING")
        elif registered_model_version and registered_model_version != MODEL_VERSION:
            reasons.add("PROFILE_MODEL_MISMATCH", registered_model_version)
        elif len(registered_embedding) != len(final_emb or registered_embedding):
            reasons.add("PROFILE_MODEL_MISMATCH", f"dim {len(registered_embedding)}")
        elif final_emb and frontal_embs:
            final_sim = SFaceRecognizer.compute_similarity(final_emb, registered_embedding)
            fused_emb = SFaceRecognizer.average_embeddings(frontal_embs, frontal_weights)
            fused_sim = SFaceRecognizer.compute_similarity(fused_emb, registered_embedding)
            # Kết hợp độ tương đồng tổng hợp đa frame nhưng yêu cầu frame FINAL cũng phải đạt ngưỡng tối thiểu
            similarity = round(max(final_sim, fused_sim), 4) if final_sim >= MIN_FRONTAL_SIMILARITY else final_sim
            matched = (
                similarity >= MIN_SIMILARITY_THRESHOLD
                and final_sim >= MIN_FRONTAL_SIMILARITY
                and final.face_count == 1
                and not reasons.has("MULTIPLE_FACES")
            )
            if not matched and not reasons.has("MULTIPLE_FACES"):
                extra_note = f"{similarity * 100:.1f}%"
                if final.sharpness is not None and final.sharpness < MIN_FACE_SHARPNESS:
                    extra_note += " – ảnh hơi mờ"
                elif final.brightness is not None and (final.brightness < MIN_FACE_BRIGHTNESS or final.brightness > MAX_FACE_BRIGHTNESS):
                    extra_note += " – ánh sáng kém"
                reasons.add("FACE_MISMATCH", extra_note)

        # 6) Chống tráo người: kiểm tra đồng nhất giữa các frame nhìn thẳng (NEUTRAL ↔ FINAL & NEUTRAL ↔ Hồ sơ)
        consistency: List[float] = []
        if final_emb and len(identity_frames) >= 2:
            for fa in identity_frames:
                if fa.index == (final.index if final else -1):
                    continue
                if fa.embedding:
                    consistency.append(SFaceRecognizer.compute_similarity(fa.embedding, final_emb))
                elif fa.face_count > 0:
                    consistency.append(0.0)
            # Mọi frame NEUTRAL cũng phải khớp hồ sơ gốc (>= MIN_FRONTAL_SIMILARITY)
            if registered_embedding:
                for fa in identity_frames:
                    if fa.tag != "NEUTRAL" or not fa.embedding:
                        continue
                    sim_n = SFaceRecognizer.compute_similarity(fa.embedding, registered_embedding)
                    if sim_n < MIN_FRONTAL_SIMILARITY:
                        consistency.append(sim_n)
        if consistency and min(consistency) < min(MIN_SESSION_CONSISTENCY, MIN_FRONTAL_SIMILARITY):
            reasons.add("IDENTITY_SWITCH", f"thấp nhất {min(consistency):.2f}")
            matched = False

        liveness_passed = anti_spoof_passed and challenge_passed
        steps_ok = sum(1 for r in step_results if r["passed"])
        details = (
            f"MiniFASNet người thật {liveness_score * 100:.0f}% ({len(frontal_scores)} frame) • "
            f"Cử động server xác minh {steps_ok}/{len(expected_actions)} • "
            f"SFace {similarity * 100:.1f}%"
        )
        if consistency:
            details += f" • Consistency frontal {min(consistency):.2f}"
        if reasons.items:
            details += " • " + " | ".join(r["message"] for r in reasons.items)

        return {
            "matched": bool(matched),
            "similarity": float(max(0.0, similarity)),
            "liveness_score": liveness_score,
            "liveness_passed": bool(liveness_passed),
            "anti_spoof_passed": bool(anti_spoof_passed),
            "challenge_passed": bool(challenge_passed),
            "model_version": MODEL_VERSION,
            "liveness_model_version": LIVENESS_MODEL_VERSION,
            "details": details,
            "reasons": reasons.items,
            "steps": step_results,
            "frames": [fa.public() for fa in analyses],
            "thresholds": {
                "similarity": MIN_SIMILARITY_THRESHOLD,
                "liveness": MIN_LIVENESS_THRESHOLD,
                "liveness_per_frame": MIN_LIVENESS_PER_FRAME,
                "consistency": MIN_SESSION_CONSISTENCY,
            },
        }

    # ------------------------------------------------------------------ enroll
    def enroll(self, images: List[bytes]) -> dict:
        reasons = Reasons()
        samples: List[FrameAnalysis] = []
        decoded_bgrs: List[Optional[np.ndarray]] = []
        for i, data in enumerate(images):
            try:
                decoded_bgrs.append(self.decode_image(data))
            except ValueError:
                decoded_bgrs.append(None)
                reasons.add("IMAGE_DECODE_FAILED", f"ảnh {i + 1}")

        with self._acquire_worker() as worker:
            for i, bgr in enumerate(decoded_bgrs):
                if bgr is None:
                    continue
                fa = self.analyze_frame(
                    worker, bgr, i, "ENROLL", 0,
                    want_hands=False, hand_multi_scale=False, want_embedding=True,
                )
                if fa.face_count == 0:
                    reasons.add("NO_FACE", f"ảnh {i + 1}")
                elif fa.face_count > 1:
                    reasons.add("MULTIPLE_FACES", f"ảnh {i + 1}")
                elif fa.face_size < MIN_FACE_SIZE_PX or not fa.embedding:
                    reasons.add("FACE_TOO_SMALL", f"ảnh {i + 1}")
                else:
                    if fa.sharpness is not None and fa.sharpness < MIN_FACE_SHARPNESS:
                        reasons.add("FACE_TOO_BLURRY", f"ảnh {i + 1}: độ nét {fa.sharpness:.0f}")
                    if fa.brightness is not None and (fa.brightness < MIN_FACE_BRIGHTNESS or fa.brightness > MAX_FACE_BRIGHTNESS):
                        reasons.add("BAD_LIGHTING", f"ảnh {i + 1}: độ sáng {fa.brightness:.0f}")
                    if fa.yaw is not None and abs(fa.yaw) > ENROLL_MAX_ABS_YAW:
                        reasons.add("POSE_TOO_EXTREME", f"ảnh {i + 1}: góc nghiêng {abs(fa.yaw):.0f}° > {ENROLL_MAX_ABS_YAW:.0f}°")
                samples.append(fa)

        # Chạy MiniFASNet theo batch cho các mẫu hợp lệ
        spoof_targets = [
            (s, decoded_bgrs[s.index], s.bbox)
            for s in samples
            if s.face_count == 1 and s.face_size >= MIN_FACE_SIZE_PX and s.bbox and decoded_bgrs[s.index] is not None
        ]
        if spoof_targets:
            spoof_results = self.anti_spoof.predict_batch([(img, bbox) for _, img, bbox in spoof_targets])
            for (s, _, _), sp in zip(spoof_targets, spoof_results):
                s.spoof = sp

        valid = [s for s in samples if s.embedding and s.face_count == 1]
        real_scores = [s.spoof["real_score"] for s in valid if s.spoof]
        liveness_score = round(float(np.mean(real_scores)), 4) if real_scores else 0.0
        if real_scores and (liveness_score < MIN_LIVENESS_THRESHOLD or min(real_scores) < MIN_LIVENESS_PER_FRAME):
            reasons.add("SPOOF_DETECTED", f"điểm người thật {liveness_score:.2f}")

        embedding: List[float] = []
        min_pair = None
        if valid:
            # Ưu tiên trọng số cao hơn (1.5) cho mẫu nhìn thẳng (|yaw| <= 12°) để tâm embedding chuẩn chính diện
            weights = [1.5 if (s.yaw is not None and abs(s.yaw) <= 12.0) else 1.0 for s in valid]
            embedding = SFaceRecognizer.average_embeddings([s.embedding for s in valid], weights)
            sims = [SFaceRecognizer.compute_similarity(s.embedding, embedding) for s in valid]
            min_pair = min(sims)
            if min_pair < 0.55:
                reasons.add("INCONSISTENT_SAMPLES", f"{min_pair:.2f}")

        success = bool(valid) and len(valid) == len(images) and not reasons.items
        return {
            "success": success,
            "model_version": MODEL_VERSION,
            "embedding": embedding if success else [],
            "embedding_dim": len(embedding) if success else 0,
            "liveness_score": liveness_score,
            "sample_consistency": min_pair,
            "reasons": reasons.items,
            "samples": [s.public() for s in samples],
        }
