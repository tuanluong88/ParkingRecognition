import cv2
import requests
import threading
import time
import csv
import os

# 카메라 설정
camera_configs = [
    {
        "name": "cam1",
        "rtsp_url": "rtsp://admin:parking1@192.168.100.30:554",
        "upload_url": "https://smell-armored-iphone.ngrok-free.dev/upload/cam1"
    },
    {
        "name": "cam2",
        "rtsp_url": "rtsp://admin:parking1@192.168.100.40:554",
        "upload_url": "https://smell-armored-iphone.ngrok-free.dev/upload/cam2"
    }
]
#CSV 파일 위치
SAVE_PATH = "/home/tuanluong/rise_parking/final_check_0610_v3"
# CSV 파일 초기화

#JETSON 내부에 csv파일 저장시 활성화(1)
for cam in camera_configs:
    csv_file = os.path.join(SAVE_PATH, f"{cam['name']}_latency.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_name" ,"request_time", "upload_ready_time", "server_received_time", "server_procesing_done_time", "JETSON_received_time", "processing_duration_sec", "JETSON-server_delay", "time_offset"])


latest_frames = {}
frame_locks = {}

#프레임 캡처 루프
def capture_loop(cam_name, rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[{cam_name}] RTSP Connection Failed")
        return

    print(f"[{cam_name}] RTSP stream connected.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_locks[cam_name]:
            latest_frames[cam_name] = frame

# 업로드 루프
def upload_loop(cam_name, upload_url):
    #JETSON 내부에 csv 저장시 활성화(2)
    csv_file = os.path.join(SAVE_PATH, f"{cam_name}_latency.csv")
    frame_id = 0
    last_sent_time = 0
    while True:
        # 1. 이미지 요청(가져오기 시작) 시간 측정
        request_start_time = time.time()
        
        frame = None
        with frame_locks[cam_name]:
            frame = latest_frames.get(cam_name, None)

        if frame is not None:
            # 2. 전처리
            resized = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            _, img_encoded = cv2.imencode('.jpg', resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_bytes = img_encoded.tobytes()

            # 3. 업로드 준비 완료 시간 측정
            upload_ready_time = time.time()
            
            # 4. 소요 시간 계산
            processing_duration = upload_ready_time - request_start_time
            

            # 서버로 보낼 타임스탬프
            timestamp_str = str(float(upload_ready_time))
            request_start_time_str = str(float(request_start_time))
            upload_ready_time_str = str(float(upload_ready_time))
            duration_str = str(float(processing_duration))
            frame_name = f"{cam_name}_{frame_id}"
            # 서버 전송 데이터
            files = {
                'image': ('frame.jpg', img_bytes, 'image/jpeg'),
                'timestamp': (None, timestamp_str),
                'request_time': (None, request_start_time_str),
                'upload_time': (None, upload_ready_time_str),
                'duration': (None, duration_str),
                'frame_name':(None, frame_name)
            }

            for attempt in range(5):
                try:
                    response = requests.post(upload_url, files=files, timeout=3)
                    print(f"[{cam_name}] Frame {frame_id} Upload Status: {response.status_code}, Timestamp: {timestamp_str}")
                    #JETSON 내부에 csv 저장 시 활성화(3)
                    t4 = time.time()
                    t1 = upload_ready_time
                    res_json = response.json()
                    t2 = res_json.get("server_received_time")
                    t3 = res_json.get("server_processing_done_time")
                    status = res_json.get("status")
                    # 편도 네트워크 지연 시간
                    delay = ((t4 - t1) - (t3 - t2)) / 2
                    # 서버와 Jetson의 시계 차이 (서버시간 - Jetson시간)
                    offset = ((t2 - t1) + (t3 - t4)) / 2
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_name, request_start_time, upload_ready_time, t2, t3, t4, processing_duration, delay, offset])
                    break
                except Exception as e:
                    print(f"[{cam_name}] Upload Error (Attempt {attempt+1}/5): {e}")
                    time.sleep(0.5)

            frame_id += 1

        elapsed_time = time.time() - last_sent_time
        sleep_time = 1.0 - elapsed_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            
        # 기준 시간을 '현재(방금 sleep이 끝난 시점)'로 갱신하여 다음 1초를 계산
        last_sent_time = time.time()

# 각 카메라 스레드 시작
for cam in camera_configs:
    name = cam["name"]
    latest_frames[name] = None
    frame_locks[name] = threading.Lock()
    threading.Thread(target=capture_loop, args=(name, cam["rtsp_url"]), daemon=True).start()
    threading.Thread(target=upload_loop, args=(name, cam["upload_url"]), daemon=True).start()

try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    print("Terminated")