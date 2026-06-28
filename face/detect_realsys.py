import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import sqlite3
from insightface.utils import face_align
import config as c

DB_PATH = c.DB_PATH

# ==========================================
# 1. DB 로드 및 식별 유틸리티
# ==========================================
def load_registered_users():
    """DB에 등록된 모든 인원의 정보와 벡터를 메모리에 불러옵니다."""
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        cursor = conn.cursor()
        cursor.execute("SELECT name, auth_group, vector FROM users")
        data = cursor.fetchall()
        conn.close()
        return data
    except Exception as e:
        print(f"❌ DB 로드 실패: {e}")
        return []

def cosine_similarity(vec1, vec2):
    v1, v2 = vec1.flatten(), vec2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# ==========================================
# 2. UI 표식 그리기 함수들
# ==========================================
def put_korean_text(img, text, position, font_size, color):
    b, g, r = color
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=(b, g, r))
    return np.array(img_pil)

def draw_green_circle(img, center, size=6):
    cv2.circle(img, center, size, (0, 255, 0), -1)

def draw_yellow_triangle(img, center, size=6):
    cx, cy = center
    pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], np.int32)
    cv2.fillPoly(img, [pts], (0, 255, 255))

def draw_red_x(img, center, size=6, thickness=2):
    cx, cy = center
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), thickness)
    cv2.line(img, (cx + size, cy - size), (cx - size, cy + size), (0, 0, 255), thickness)

# ==========================================
# 3. 메인 파이프라인용 얼굴 식별 및 표식 부착
# ==========================================
def process_face_recognition(frame, bboxes, kpss, recognizer, db_users):
    """
    넉넉한 영역을 블러 처리한 뒤, 그 겉면에 박스와 텍스트를 표시합니다.
    """
    blur_bboxes = []
    blur_kpss = []

    if bboxes is not None and len(bboxes) > 0:
        for i in range(bboxes.shape[0]):
            x1, y1, x2, y2 = bboxes[i, 0:4].astype(int)
            landmarks = kpss[i]

            # 1. 블러 덮어쓰기 전에 원본 얼굴로 AI 특징 추출!
            face_aligned = face_align.norm_crop(frame, landmark=landmarks, image_size=112)
            embedding = recognizer.get_feat(face_aligned)

           # 2. 💡 [최종 진화] 윗머리 보장 + 고개 각도에 따른 '부드러운' 스마트 마진
            face_w = x2 - x1
            face_h = y2 - y1
            
            # 윗머리 여백 복구 (35%로 늘려서 정수리 넉넉하게 덮기), 턱은 타이트하게 5%
            margin_top = int(face_h * 0.35)    
            margin_bottom = int(face_h * 0.05) 
            
            # 코의 위치를 0.0 ~ 1.0 사이의 비율로 계산 (0.5면 완벽한 정면)
            nose_x = landmarks[2][0]
            nose_ratio = (nose_x - x1) / face_w
            
            # 기본 좌우 여백 (정면일 때 뚱뚱해지지 않게 25%만)
            base_margin_x = int(face_w * 0.25) 
            
            # 뒷통수 쪽으로 늘어날 최대 여백 한도
            max_extra_margin = face_w * 0.6

            # 💡 [핵심] 코 방향의 '반대편(뒤통수)'이 늘어나도록 좌우 수정!
            if nose_ratio < 0.4:  
                # 코가 화면 '왼쪽'으로 쏠림 -> 뒷통수는 화면 '오른쪽'에 있음! (margin_right 팽창)
                margin_left = base_margin_x
                margin_right = base_margin_x + int(max_extra_margin * ((0.4 - nose_ratio) / 0.4))
            elif nose_ratio > 0.6: 
                # 코가 화면 '오른쪽'으로 쏠림 -> 뒷통수는 화면 '왼쪽'에 있음! (margin_left 팽창)
                margin_left = base_margin_x + int(max_extra_margin * ((nose_ratio - 0.6) / 0.4))
                margin_right = base_margin_x
            else: 
                # 코가 중앙 부근(0.4 ~ 0.6)에 있음 -> 완벽한 정면! 양쪽 동일하게 유지
                margin_left = base_margin_x
                margin_right = base_margin_x

            # 최종 좌표 계산
            mx1 = max(0, int(x1 - margin_left))
            my1 = max(0, int(y1 - margin_top))
            mx2 = min(frame.shape[1], int(x2 + margin_right))
            my2 = min(frame.shape[0], int(y2 + margin_bottom))

            # 3. 💡 넉넉한 영역 안에 가우시안 블러 먼저 칠하기!
            if mx2 > mx1 and my2 > my1:
                face_roi = frame[my1:my2, mx1:mx2]
                face_roi = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[my1:my2, mx1:mx2] = face_roi

            # 4. DB 비교를 통해 누구인지 판별
            best_group, max_sim = "비허가", -1
            if embedding is not None and db_users:
                for db_name, db_group, db_vector in db_users:
                    sim = cosine_similarity(embedding, db_vector)
                    if sim > max_sim:
                        max_sim, best_group = sim, db_group

            # 5. 💡 블러가 끝난 '넉넉한 박스' 테두리에 선과 글씨 그리기
            if best_group in ["허가", "준허가"] and max_sim > c.MATCH_THRESHOLD:
                if best_group == "허가":
                    color, marker_func, text = (0, 255, 0), draw_green_circle, "허가자"
                else:
                    color, marker_func, text = (0, 255, 255), draw_yellow_triangle, "준허가자"
                
                # 기존 (x1, y1)이 아닌 넓어진 (mx1, my1) 기준!
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 2)
                ui_y = max(my1 - 10, 20)
                marker_func(frame, (mx1 + 10, ui_y - 4))
                frame = put_korean_text(frame, text, (mx1 + 25, ui_y - 15), 20, color)
            else:
                cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 0, 255), 2)
                frame = put_korean_text(frame, "비허가자", (mx1 + 25, max(my1 - 10, 20) - 15), 20, (0, 0, 255))

            # 보호 캡처(INN)용으로는 원래 사이즈 좌표를 넘김
            blur_bboxes.append(bboxes[i])
            blur_kpss.append(kpss[i])

    return frame, blur_bboxes, blur_kpss