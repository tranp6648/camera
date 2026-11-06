import json
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import (
    FACE_AI_TOKEN, LIVENESS_MODEL_VERSION, MAX_FRAMES, MAX_IMAGE_BYTES, MODEL_VERSION, PORT,
)
from .pipeline import FaceVerificationPipeline, FrameMeta

logger = logging.getLogger("face-ai")

# Chỉ bật khi cần chẩn đoán (ảnh sinh trắc học!) – lưu frame + kết quả phân tích mỗi request
DEBUG_DUMP_DIR = os.getenv("FACE_DEBUG_DUMP_DIR", "").strip()


def _dump(kind: str, employee_id: str, images: List[bytes], result: dict, extra: Optional[dict] = None) -> None:
    if not DEBUG_DUMP_DIR:
        return
    try:
        d = Path(DEBUG_DUMP_DIR) / f"{time.strftime('%Y%m%d-%H%M%S')}_{kind}_{employee_id or 'na'}"
        d.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(images):
            (d / f"{i:02d}.jpg").write_bytes(data)
        (d / "result.json").write_text(json.dumps({**result, "embedding": None, **(extra or {})},
                                                  ensure_ascii=False, indent=1, default=str))
    except Exception as exc:  # debug không được làm hỏng request
        logger.warning("debug dump failed: %s", exc)

app = FastAPI(
    title="Internal Face AI Microservice",
    description="Face detection (YuNet), recognition (SFace), anti-spoofing (MiniFASNet) "
                "and server-side liveness challenge verification (MediaPipe)",
    version="3.0.0",
)

# Dịch vụ chỉ được gọi nội bộ bởi NestJS -> không bật CORS cho trình duyệt
pipeline = FaceVerificationPipeline()


def verify_internal_auth(x_internal_token: str = Header(default="")):
    if not x_internal_token or x_internal_token != FACE_AI_TOKEN:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED_INTERNAL_SERVICE: Invalid service token")
    return True


async def _read_images(files: List[UploadFile]) -> List[bytes]:
    if not files:
        raise HTTPException(status_code=422, detail="EMPTY_IMAGE_PAYLOAD")
    if len(files) > MAX_FRAMES:
        raise HTTPException(status_code=413, detail="TOO_MANY_FRAMES")
    out = []
    for f in files:
        data = await f.read()
        if not data:
            raise HTTPException(status_code=422, detail="EMPTY_IMAGE_PAYLOAD")
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="IMAGE_TOO_LARGE")
        out.append(data)
    return out


def _parse_json(raw: Optional[str], default):
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="INVALID_JSON_FIELD")


@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "face-ai-service",
        "models": {
            "detection": "yunet_2023mar",
            "recognition": MODEL_VERSION,
            "anti_spoof": LIVENESS_MODEL_VERSION,
            "landmarks": "mediapipe_face_landmarker+hand_landmarker",
        },
    }


@app.post("/internal/face/enroll", dependencies=[Depends(verify_internal_auth)])
async def enroll_face(images: List[UploadFile] = File(...), employee_id: Optional[str] = Form("")):
    """Đăng ký khuôn mặt từ 1-5 ảnh mẫu: kiểm tra 1 khuôn mặt, chống giả mạo, trả embedding trung bình."""
    if len(images) > 5:
        raise HTTPException(status_code=413, detail="TOO_MANY_SAMPLES")
    data = await _read_images(images)
    result = await run_in_threadpool(pipeline.enroll, data)
    _dump("enroll", employee_id or "", data, result)
    return result


@app.post("/internal/face/verify", dependencies=[Depends(verify_internal_auth)])
async def verify_face(
    frames: List[UploadFile] = File(...),
    frames_meta_json: str = Form(...),
    expected_actions_json: str = Form("[]"),
    registered_embedding_json: Optional[str] = Form(None),
    registered_model_version: Optional[str] = Form(None),
    employee_id: Optional[str] = Form(""),
):
    """Xác minh 1 phiên chấm công từ chuỗi frame thử thách (gọi bởi NestJS CRM Backend)."""
    data = await _read_images(frames)
    raw_meta = _parse_json(frames_meta_json, [])
    if not isinstance(raw_meta, list) or len(raw_meta) != len(data):
        raise HTTPException(status_code=422, detail="FRAMES_META_MISMATCH")
    try:
        metas = [FrameMeta(tag=str(m["tag"]), step=int(m.get("step", -1)), ts=int(m["ts"])) for m in raw_meta]
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=422, detail="INVALID_FRAME_META")

    expected_actions = _parse_json(expected_actions_json, [])
    registered_embedding = _parse_json(registered_embedding_json, None)
    if registered_embedding is not None and not isinstance(registered_embedding, list):
        registered_embedding = None

    result = await run_in_threadpool(
        pipeline.verify_session, data, metas, expected_actions, registered_embedding, registered_model_version,
    )
    logger.info(
        "verify employee=%s matched=%s sim=%.3f live=%.3f spoof_ok=%s challenge_ok=%s reasons=%s",
        employee_id, result["matched"], result["similarity"], result["liveness_score"],
        result["anti_spoof_passed"], result["challenge_passed"], [r["code"] for r in result["reasons"]],
    )
    _dump("verify", employee_id or "", data, result, {"metas": raw_meta, "expected_actions": expected_actions})
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
