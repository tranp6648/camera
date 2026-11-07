import os
from pathlib import Path

FACE_AI_TOKEN = os.getenv("FACE_AI_TOKEN", "crm_internal_face_secret_token_2026")
PORT = int(os.getenv("PORT", "8000"))

# Thư mục chứa file model (docker-compose mount ./face-ai-service/models -> /app/models)
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")))

# --- Phiên bản model (lưu cùng embedding để phát hiện hồ sơ cũ cần đăng ký lại) ---
MODEL_VERSION = "opencv_sface_2021dec"           # Embedding 128-dim (OpenCV Zoo SFace, Apache-2.0)
LIVENESS_MODEL_VERSION = "minifasnet_v2+v1se"    # Silent-Face-Anti-Spoofing (Apache-2.0)

# --- Ngưỡng quyết định (có thể chỉnh qua biến môi trường khi hiệu chỉnh thực tế) ---
# SFace cosine: OpenCV khuyến nghị 0.363; dùng 0.42 để chặt hơn cho chấm công 1:1
MIN_SIMILARITY_THRESHOLD = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.42"))
# Ngưỡng tối thiểu cho từng frame nhìn thẳng riêng lẻ so với hồ sơ gốc
MIN_FRONTAL_SIMILARITY = float(os.getenv("MIN_FRONTAL_SIMILARITY", "0.38"))
# Ngưỡng tối thiểu giữa các frame NHÌN THẲNG (NEUTRAL↔FINAL) trong phiên (chống tráo người).
# Không áp dụng cho frame ACTION (quay đầu/giơ tay) vì SFace tụt mạnh dù cùng người.
MIN_SESSION_CONSISTENCY = float(os.getenv("MIN_SESSION_CONSISTENCY", "0.40"))
# Xác suất "người thật" trung bình (MiniFASNet) trên các frame nhìn thẳng
# CẦN HIỆU CHỈNH với webcam thực tế (ghi log real_score của người thật + ảnh in/màn hình rồi chọn ngưỡng)
MIN_LIVENESS_THRESHOLD = float(os.getenv("MIN_LIVENESS_THRESHOLD", "0.65"))
# Không frame nhìn thẳng nào được thấp hơn mức này
MIN_LIVENESS_PER_FRAME = float(os.getenv("MIN_LIVENESS_PER_FRAME", "0.45"))

MIN_FACE_SIZE_PX = int(os.getenv("MIN_FACE_SIZE_PX", "64"))
FACE_DETECT_SCORE = float(os.getenv("FACE_DETECT_SCORE", "0.70"))

# --- Chất lượng ảnh khuôn mặt & giới hạn góc khi đăng ký (Enrollment) ---
MIN_FACE_SHARPNESS = float(os.getenv("MIN_FACE_SHARPNESS", "18.0"))
MIN_FACE_BRIGHTNESS = float(os.getenv("MIN_FACE_BRIGHTNESS", "30.0"))
MAX_FACE_BRIGHTNESS = float(os.getenv("MAX_FACE_BRIGHTNESS", "240.0"))
# Cho phép nghiêng mạnh / gần nghiêng hết khi đăng ký (UI client hay bắt ~40–50°)
ENROLL_MAX_ABS_YAW = float(os.getenv("ENROLL_MAX_ABS_YAW", "60.0"))

# Số lượng worker instance xử lý song song trong pipeline (tránh nghẽn cổ chai 1 lock)
FACE_AI_WORKERS = max(1, int(os.getenv("FACE_AI_WORKERS", "2")))

MAX_FRAMES = 12
MAX_IMAGE_BYTES = 600 * 1024
MAX_IMAGE_SIDE = 960

# --- Ngưỡng hành động (server luôn nới hơn client một chút để tránh client đạt mà server trượt) ---
NEUTRAL_MAX_ABS_YAW = 12.0
TURN_MIN_ABS_YAW = 16.0
BLINK_CLOSED_SCORE = 0.45      # blendshape eyeBlink trung bình khi nhắm
BLINK_OPEN_MAX_SCORE = 0.35    # blendshape eyeBlink khi mở
BLINK_EAR_RATIO = 0.75         # EAR lúc nhắm < 75% EAR lúc mở
MIN_ACTION_GAP_MS = 120        # frame hành động phải cách frame trung tính tối thiểu
MAX_SESSION_SPAN_MS = 120_000
