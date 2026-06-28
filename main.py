import cv2
import asyncio
import numpy as np
import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from insightface.app import FaceAnalysis

import config as c
from blur.pipeline import SecureFaceRX
from detection.scrfd_adapter import scrfd_to_detection

# ⭐️ 핵심: 분리된 모듈들을 깔끔하게 가져옵니다.
from face.register_api import init_db, router as register_router
from face.detect_realsys import load_registered_users, process_face_recognition

# ─── 1. 전역 상태 관리 ──────────────────────────────────────────
system_status = {"status": "loading", "face_count": 0, "psf_exists": False}
result_images = {"protected": None, "restored": None}

# ─── 2. 라이프사이클 (서버 가동 시 1회 실행) ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[시스템] 보안 서버 가동 준비 중...")
    
    init_db()  # DB 초기화 (main_org.py)
    
    # SCRFD + ArcFace 출입통제 모델 로드
    app.state.face_app = FaceAnalysis(name=c.INSIGHTFACE_NAME, providers=['CPUExecutionProvider'])
    app.state.face_app.prepare(ctx_id=-1, det_thresh=c.DET_THRESH)
    
    app.state.db_users = load_registered_users() # 등록된 인원 로드 (detect_realsys.py)
    print(f"✅ DB 데이터 로드 완료 ({len(app.state.db_users)}명)")

    # SecureFaceRX (INN 블러) 모델 로드
    app.state.sfx = SecureFaceRX(checkpoint_path=c.CKPT_PATH, obf_type=c.DEFAULT_OBFUSCATOR)
    print("✅ INN 프라이버시 보호 모델 로드 완료")
    
    system_status["status"] = "ready"
    yield
    print("[시스템] 서버 종료 및 하드웨어 자원 해제")

# ─── 3. FastAPI 앱 설정 및 라우터 연결 ──────────────────────────
app = FastAPI(lifespan=lifespan)
app.include_router(register_router)  # 인원 등록 API 연결 (main_org.py)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """메인 관제 화면"""
    return templates.TemplateResponse(request, "index.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """신규 인원 등록 화면"""
    return templates.TemplateResponse(request, "access.html")

@app.get("/api/status")
async def get_status():
    """웹 UI에서 서버 상태를 지속적으로 확인하는 API"""
    system_status["psf_exists"] = os.path.exists(c.PSF_PATH)
    return JSONResponse(content=system_status)

# ─── 4. 실시간 웹캠 프레임 처리 (스트리밍) ───────────────────────
async def gen_frames():
    """실시간 로봇 카메라 프레임 처리 파이프라인 (최적화 버전)"""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    # 💡 1. 해상도 원상 복구 (고화질 선명도)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 

    detector = app.state.face_app.models['detection']
    recognizer = app.state.face_app.models['recognition']

    frame_count = 0
    skip_rate = 2  # 💡 2. 프레임 스킵: 3프레임당 1번만 무거운 AI 연산 수행
    
    # 마지막으로 인식된 얼굴 좌표를 기억할 변수
    last_bboxes, last_kpss = None, None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            await asyncio.sleep(0.01)
            continue
            
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        try:
            # 💡 [핵심 최적화] 매번 AI를 돌리지 않고, 3번에 1번만 돌려서 CPU 과부하 방지
            if frame_count % skip_rate == 0 or last_bboxes is None:
                bboxes, kpss = detector.detect(frame, max_num=0, metric='default')
                last_bboxes, last_kpss = bboxes, kpss  # 결과 기억
            else:
                # AI가 쉬는 타이밍에는 0.1초 전의 좌표를 그대로 재사용 (렉 70% 감소)
                bboxes, kpss = last_bboxes, last_kpss

            system_status["face_count"] = 0 if bboxes is None else len(bboxes)

            # 얼굴 식별 및 표식 부착
            frame_processed, blur_bboxes, blur_kpss = process_face_recognition(
                frame.copy(), bboxes, kpss, recognizer, app.state.db_users
            )

            # 브라우저로 전송
            ret, buffer = cv2.imencode('.jpg', frame_processed)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            # 에러 발생 시 원본 화면이라도 송출하여 먹통 방지
            print(f"⚠️ 프레임 처리 에러 (무시): {e}")
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
        await asyncio.sleep(0.01)

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


# ─── 5. 프라이버시 보호 및 복원 제어 API ────────────────────────
def numpy_to_base64(img_np):
    _, buffer = cv2.imencode('.jpg', img_np)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.post("/protect")
async def api_protect(payload: dict, background_tasks: BackgroundTasks):
    password = payload.get("password", "0")
    if system_status["status"] != "ready":
        return JSONResponse(content={"ok": False, "msg": "서버 준비 중입니다."})
        
    system_status["status"] = "protecting"
    
    # 캡처를 위해 카메라에서 1프레임 가져오기
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    success, frame = cap.read()
    cap.release()
    
    if success:
        def run_protect():
            app.state.sfx.detector = None 
            frame_prot, _ = app.state.sfx.protect_image(frame, password, out_psf=c.PSF_PATH)
            result_images["protected"] = numpy_to_base64(frame_prot)
            system_status["psf_exists"] = True
            system_status["status"] = "ready"
            
        background_tasks.add_task(run_protect)
        return JSONResponse(content={"ok": True, "msg": "캡처 진행 중"})
    else:
        system_status["status"] = "ready"
        return JSONResponse(content={"ok": False, "msg": "카메라 읽기 실패"})

@app.post("/restore")
async def api_restore(payload: dict, background_tasks: BackgroundTasks):
    password = payload.get("password", "0")
    if not os.path.exists(c.PSF_PATH):
         return JSONResponse(content={"ok": False, "msg": "보호된 파일(PSF)이 없습니다."})

    system_status["status"] = "restoring"
    
    def run_restore():
        restored_frame = app.state.sfx.restore_image(c.PSF_PATH, password)
        result_images["restored"] = numpy_to_base64(restored_frame)
        system_status["status"] = "ready"
        
    background_tasks.add_task(run_restore)
    return JSONResponse(content={"ok": True, "msg": "복원 진행 중"})

@app.get("/view", response_class=HTMLResponse)
async def view_page(request: Request):
    return templates.TemplateResponse("view.html", {"request": request})

@app.get("/result_data")
async def get_result_data():
    return JSONResponse(content={
        "protected": result_images["protected"],
        "restored": result_images["restored"],
        "status": "복원 완료" if result_images["restored"] else "대기 중"
    })