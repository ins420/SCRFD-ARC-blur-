import os
import sqlite3
import base64
import io
import cv2
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from insightface.utils import face_align
import config as c

# 설정값 불러오기
IMAGE_DIR = c.IMAGE_DIR
DB_PATH = c.DB_PATH
os.makedirs(IMAGE_DIR, exist_ok=True)

# ==========================================
# 1. SQLite <-> Numpy 변환 어댑터 및 초기화
# ==========================================
def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def init_db():
    """서버 시작 시 데이터베이스가 없으면 생성합니다."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            auth_group TEXT NOT NULL,
            image_path TEXT NOT NULL,
            vector array NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# ==========================================
# 2. 얼굴 등록 API 라우터
# ==========================================
router = APIRouter()

class RegisterData(BaseModel):
    name: str
    group: str
    image_base64: str

@router.post("/api/register")
async def register_person(data: RegisterData, request: Request):
    try:
        # 통합 main.py에서 메모리에 올려둔 모델을 끌어다 씁니다 (중복 로드 방지)
        face_app = request.app.state.face_app
        detector = face_app.models['detection']
        recognizer = face_app.models['recognition']

        # 이미지 디코딩
        header, encoded = data.image_base64.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        bboxes, kpss = detector.detect(frame, max_num=1, metric='default')

        if bboxes is None or len(bboxes) == 0:
            return JSONResponse(content={"status": "error", "message": "❌ 사진에서 얼굴을 찾을 수 없습니다."})

        x1, y1, x2, y2 = int(bboxes[0, 0]), int(bboxes[0, 1]), int(bboxes[0, 2]), int(bboxes[0, 3])
        landmarks = kpss[0]

        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        dist_left = np.linalg.norm(left_eye - nose)
        dist_right = np.linalg.norm(right_eye - nose)
        ratio = max(dist_left, dist_right) / (min(dist_left, dist_right) + 1e-5)
        box_width = x2 - x1
        eye_dist = np.linalg.norm(left_eye - right_eye)
        is_extreme_side = (eye_dist / box_width) < 0.25
        is_side_face = (ratio > 1.5) or is_extreme_side

        face_aligned = face_align.norm_crop(frame, landmark=landmarks, image_size=112)
        embedding = recognizer.get_feat(face_aligned)

        if embedding is None:
            return JSONResponse(content={"status": "error", "message": "❌ 얼굴 특징 벡터를 추출할 수 없습니다."})

        if is_side_face:
            if dist_left < dist_right:
                file_name, msg_tag = f"{data.name}_측면1(좌).jpg", "좌측면"
            else:
                file_name, msg_tag = f"{data.name}_측면2(우).jpg", "우측면"
        else:
            file_name, msg_tag = f"{data.name}_정면.jpg", "정면"

        file_path = os.path.join(IMAGE_DIR, file_name)
        cv2.imwrite(file_path, frame)

        # DB 저장
        db_store_name = f"{data.name}_{msg_tag}"
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, auth_group, image_path, vector) VALUES (?, ?, ?, ?)",
            (db_store_name, data.group, file_path, embedding)
        )
        conn.commit()
        conn.close()

        # ⭐️ 새로운 인원이 등록되면, 서버의 DB 메모리를 즉시 새로고침하여 바로 인식되게 함
        from face.detect_realsys import load_registered_users
        request.app.state.db_users = load_registered_users()

        return JSONResponse(content={"status": "success", "message": f"✅ [{data.name}]님의 [{msg_tag}] 얼굴 등록 성공!"})
    
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": f"서버 오류: {str(e)}"})