"""
SecureFace-RX 전역 설정
실제 ProFace S config/config.py 기준으로 작성
"""

import os
import torch

# ─── 디바이스 ─────────────────────────────────────────────────────
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ─── INN 아키텍처 ─────────────────────────────────────────────────
INV_BLOCKS   = 3       # INV_block_affine 반복 수 (config.INV_BLOCKS)
channels_in  = 3       # 입력 채널 수 (RGB)
clamp        = 2.0     # affine 스케일 클램핑 계수

# ─── 오복원(Wrong Recovery) 모드 ──────────────────────────────────
# 'Random': RandWR — 랜덤 노이즈형 오복원 (PSNR<11dB)
# 'Obfs'  : ObfsWR — 난독화 유지형 오복원
WRONG_RECOVER_TYPE = 'Random'

# ─── 키 보조입력 정책 ─────────────────────────────────────────────
SECRET_KEY_AS_NOISE = True  # 복원 보조입력으로 K를 3채널 반복

# ─── Utility 조건부 기능 (기본 비활성) ───────────────────────────
ADJ_UTILITY = False

# ─── 정규화 해상도 ────────────────────────────────────────────────
# 원본 config: cropsize=224, SRS: NORM_RESOLUTION=256
# 공식 가중치 사용 시 학습된 해상도에 맞춰야 함
NORM_RESOLUTION = 256   # 변경 시 key 길이도 달라짐

# ─── 사전 난독화 ──────────────────────────────────────────────────
DEFAULT_OBFUSCATOR = 'blur'
BLUR_KERNEL_SIZE   = 61
BLUR_SIGMA         = 21.0     # 원본 hybridAll: Blur(61, 9, 21)
BLUR_SIGMA_MIN     = 9.0      # 원본 hybridAll blur sigma_min
PIXELATE_BLOCK     = 20       # 원본 hybridAll: Pixelate(20)
MEDIAN_KERNEL      = 23       # 원본 hybridAll: MedianBlur(23)

# ─── 검출기 ───────────────────────────────────────────────────────
DETECTOR_CONF_THRESHOLD = 0.25
DETECTOR_NMS_IOU        = 0.4
FACE_MARGIN             = 0.10

# ─── 학습 하이퍼파라미터 (SRS §7 / 원본 config 기준) ─────────────
lr           = 0.00001
batch_size   = 6
weight_decay = 1e-5
init_scale   = 0.01
TRIPLET_MARGIN         = 1.2
LAMBDA_RECONSTRUCTION  = 5
LAMBDA_GUIDE           = 1
LAMBDA_LOW_FREQUENCY   = 1

SAVE_IMAGE_INTERVAL = 1000
SAVE_MODEL_INTERVAL = 5000

# ─── 사전학습 가중치 파일명 ───────────────────────────────────────
CHECKPOINT_ID = "hybridAll_inv3_recTypeRandom_secretAsNoise_TripMargin1.2_ep12_iter15000"

# ─── KeyGen (PBKDF2) ── NFR-SEC-2 경고 ───────────────────────────
# !! salt=1, count=10 은 논문의 "demonstration only" 값 !!
# 운영 배포 시 임의 salt + OWASP 권고 반복 수(≥600000)로 교체할 것
KEY_SALT  = 1
KEY_COUNT = 10

# ─── 기타 ─────────────────────────────────────────────────────────
debug = False
recognizer = 'AdaFaceIR100'

# ─── 여기서부터 통합 서버 및 얼굴 인식(Access)을 위해 추가된 설정입니다 ───

# ─── 서버 기본 설정 ─────────────────────────────────
HOST = '0.0.0.0'
PORT = 8000

# ─── 얼굴 인식 및 출입 통제 (access) 설정 ───────────────────
DB_PATH          = 'db/security_system.db'
IMAGE_DIR        = 'registered_faces'
MATCH_THRESHOLD  = 0.45
INSIGHTFACE_NAME = 'buffalo_s'
DET_THRESH      = 0.6

# ─── 통합 파이프라인 작동을 위한 추가 변수 ──────────────────
# 위에서 정의된 CHECKPOINT_ID를 활용하여 실제 가중치 파일 경로 완성
CKPT_PATH = f"checkpoints/{CHECKPOINT_ID}.pth"
PSF_PATH  = 'web_output/capture.psf'

# 필수 디렉토리가 없으면 서버 시작 시 자동 생성
os.makedirs('web_output', exist_ok=True)
os.makedirs('registered_faces', exist_ok=True)
os.makedirs('db', exist_ok=True)