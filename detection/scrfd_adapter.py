"""
YOLOv8-face 검출기 플러그인 (FR-1.1, P0-1)
공식 구현의 MTCNN을 대체.
YOLOv8-face 가중치: yolov8n-face.pt (기존 blurOnOff.py 와 동일 모델 재사용)

면적 확장(FR-2.1 margin): bbox를 FACE_MARGIN 비율로 정사각형 확장 후
복원에 쓸 crop_box를 정규화 해상도(256×256)로 리사이즈.
"""

import numpy as np
import config
from detection.detector_base import DetectionResult
import cv2

# --- 1. 신규 추가: SCRFD 결과를 DetectionResult로 변환 ---
def scrfd_to_detection(bbox_arr, kps_arr) -> list[DetectionResult]:
    """
    SCRFD의 출력(bbox, landmarks)을 INN 파이프라인 호환 규격으로 변환
    """
    results = []
    # 탐지된 얼굴이 없는 경우 처리
    if bbox_arr is None or len(bbox_arr) == 0:
        return results

    for i in range(len(bbox_arr)):
        x1, y1, x2, y2 = bbox_arr[i, :4].astype(int)
        conf = float(bbox_arr[i, 4])
        
        # 랜드마크가 제공된 경우
        lm = kps_arr[i] if kps_arr is not None else None
            
        results.append(DetectionResult(
            bbox=[x1, y1, x2, y2], conf=conf, landmarks=lm
        ))
    return results

# ── 얼굴 정렬 유틸 ─────────────────────────────────────────────────

def expand_bbox_square(
    bbox: list[int],
    frame_h: int,
    frame_w: int,
    margin: float = config.FACE_MARGIN,
) -> list[int]:
    """
    bbox [x1,y1,x2,y2] → 정사각형 확장 crop_box [x1,y1,x2,y2]
    - margin: 각 변을 비율로 확장
    - 정사각형화: 긴 쪽 기준
    반환: 클리핑된 [x1,y1,x2,y2]
    """
    x1, y1, x2, y2 = bbox
    fw = x2 - x1
    fh = y2 - y1

    # margin 확장
    x1 = x1 - int(margin * fw)
    y1 = y1 - int(margin * fh)
    x2 = x2 + int(margin * fw)
    y2 = y2 + int(margin * fh)

    # 정사각형화
    new_w = x2 - x1
    new_h = y2 - y1
    side  = max(new_w, new_h)
    cx    = (x1 + x2) // 2
    cy    = (y1 + y2) // 2
    x1    = cx - side // 2
    y1    = cy - side // 2
    x2    = x1 + side
    y2    = y1 + side

    # 프레임 경계 클리핑
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    return [x1, y1, x2, y2]


def crop_and_resize(
    frame: np.ndarray,
    crop_box: list[int],
    target_size: int = config.NORM_RESOLUTION,
) -> tuple[np.ndarray, float]:
    """
    crop_box 영역을 잘라 target_size×target_size 로 리사이즈.
    반환: (face_img HWC BGR, scale)  — scale = target_size / crop_side
    """
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    import cv2
    # 다운샘플링이면 INTER_AREA (aliasing 최소화), 업샘플링이면 INTER_LINEAR
    interp = cv2.INTER_AREA if (x2 - x1) > target_size else cv2.INTER_LINEAR
    face = cv2.resize(crop, (target_size, target_size), interpolation=interp)
    scale = target_size / max(x2 - x1, y2 - y1, 1)
    return face, scale


def paste_back(
    frame: np.ndarray,
    face_img: np.ndarray,
    crop_box: list[int],
) -> np.ndarray:
    """
    face_img (target_size×target_size BGR) 를 원래 crop_box 위치에 붙여넣기.
    frame은 in-place 수정되지 않도록 복사본을 반환한다.
    """
    import cv2
    x1, y1, x2, y2 = crop_box
    side = x2 - x1
    resized = cv2.resize(face_img, (side, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out = frame.copy()
    out[y1:y2, x1:x2] = resized
    return out
