import cv2
import torch
import json
import numpy as np
import os
import time
import requests
import argparse
import cgi
import csv
from datetime import datetime, timezone
from multiprocessing import Queue
from queue import Queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import RTDETR

# ================== Setup and Paths ================== #
#lambda_url 경로
LAMBDA_URL = "https://mtm6mthy2f5yqlh7k265qnq6ba0hphhj.lambda-url.ap-southeast-2.on.aws/update-button-status"
MIN_CHANGE_FRAMES = 8      
SEND_INTERVAL = 0.5        
# RT-DETR 모델 경로
MODEL_PATH = "/home/tuanluong/coop2026_1/GPUservertest/rtdetr1000.pt"
# 이미지 저장 경로
OUTPUT_PATH = "/home/tuanluong/coop2026_1/GPUservertest/accuracy_measurement_frames"
# 주차장 점유 확인 점 위치 json 파일 경로
JSON_PATH_ANGLE_1 = "/home/tuanluong/coop2026_1/GPUservertest/json/json/parking_spots_zoneA_cal.json"
JSON_PATH_ANGLE_2 = "/home/tuanluong/coop2026_1/GPUservertest/json/json/parking_spots_zoneB_cal.json"
#latency csv 파일 경로
JETSON_LATENCY_1='/home/tuanluong/coop2026_1/GPUservertest/latency_package/Jetson_latency_cam1.csv'
JETSON_LATENCY_2='/home/tuanluong/coop2026_1/GPUservertest/latency_package/Jetson_latency_cam2.csv'
LATENCY_1="/home/tuanluong/coop2026_1/GPUservertest/latency_package/latency_cam1.csv"
LATENCY_2="/home/tuanluong/coop2026_1/GPUservertest/latency_package/latency_cam2.csv"
LAMBDA_CSV_PATHS_CAM1="/home/tuanluong/coop2026_1/GPUservertest/latency_package/latency_lambda1.csv"
LAMBDA_CSV_PATHS_CAM2="/home/tuanluong/coop2026_1/GPUservertest/latency_package/latency_lambda2.csv"
HOST = "0.0.0.0"
PORT = 8080

last_combined_send_time = time.time()
last_sent_time_dict = {"camera_1": 0, "camera_2": 0}

with open(JSON_PATH_ANGLE_1) as f:
    parking_spots_angle_1 = json.load(f)
with open(JSON_PATH_ANGLE_2) as f:
    parking_spots_angle_2 = json.load(f)

stable_state_dict = {
    "camera_1": {},
    "camera_2": {}
}
change_count_dict = {
    "camera_1": {},
    "camera_2": {}
}
first_frame_processed_dict = {
    "camera_1": False,
    "camera_2": False
}
parking_spots = {
    "camera_1": parking_spots_angle_1,
    "camera_2": parking_spots_angle_2
}
processing_info = {
    "camera_1": {"processing_time": 0, "server_sent_time": 0, "start_time": 0, "lambda_send_time":0},
    "camera_2": {"processing_time": 0, "server_sent_time": 0, "start_time": 0, "lambda_send_time":0}
}

# Load RT-DETR Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = RTDETR(MODEL_PATH).to(device) # RTDETR 클래스 사용
class_names = model.model.names

def get_unique_filename(base_path):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename, ext = os.path.splitext(base_path)
    filename_with_timestamp = f"{filename}_{timestamp}"
    if not os.path.exists(f"{filename_with_timestamp}{ext}"):
        return f"{filename_with_timestamp}{ext}"
    counter = 1
    while os.path.exists(f"{filename_with_timestamp}_{counter}{ext}"):
        counter += 1
    return f"{filename_with_timestamp}_{counter}{ext}"

def convert_stable_state_to_array(camera_id):
    state_dict = stable_state_dict[camera_id]
    spots = parking_spots[camera_id]
    # spot_id 기반 매핑 유지
    state_array = [state_dict.get(spot["id"], 0) for spot in spots]
    return state_array

def send_combined_to_lambda(processing_info, camera_id):
    global last_combined_send_time
    current_time = time.time()
    for cam in ["camera_1", "camera_2"]:
        processing_info[cam]["lambda_send_time"] = current_time

    #if (current_time - last_sent_time_dict["camera_1"] >= SEND_INTERVAL) or \
    #   (current_time - last_sent_time_dict["camera_2"] >= SEND_INTERVAL):
    

    combined_payload = {
        "button_status": {
            "camera_1": convert_stable_state_to_array("camera_1"),
            "camera_2": convert_stable_state_to_array("camera_2")
        },
        "latency_info": {
            "camera_1": processing_info["camera_1"],
            "camera_2": processing_info["camera_2"]
        }
    }
    combined_payload["send_camera_id"]=camera_id
    try:
        response = requests.post(LAMBDA_URL, json=combined_payload, timeout=2)
        res_data = response.json()
        if response.status_code == 200:
            v4 = time.time()
            print(f"[Lambda] Combined update successful.")
            v1 = current_time
            v2 = res_data.get("lambda_receive_time")
            v3 = res_data.get("lambda_finish_time")
            processing_duration = res_data.get("processing")
            frame_name = res_data.get("frame_name")
            # 1. 네트워크 지연 시간 (Network Delay / 2)
            lambda_network_delay = ((v4 - v1) - (v3 - v2)) / 2

            # 2. 서버와 람다 간의 시계 오차 (Clock Offset)
            lambda_clock_offset = ((v2 - v1) + (v3 - v4)) / 2
            total_lambda_latency=lambda_network_delay+processing_duration
            with open(LAMBDA_CSV_PATHS[camera_id], "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_name, lambda_network_delay, processing_duration, total_lambda_latency, lambda_clock_offset])
    except Exception as e:
        print(f"[Lambda] Send error: {e}")
    last_sent_time_dict["camera_1"] = current_time
    last_sent_time_dict["camera_2"] = current_time

def initialize_camera(camera_id):
    if camera_id not in stable_state_dict:
        stable_state_dict[camera_id] = {}
    if camera_id not in change_count_dict:
        change_count_dict[camera_id] = {}
    if camera_id not in first_frame_processed_dict:
        first_frame_processed_dict[camera_id] = False

initialize_camera('camera_1')
initialize_camera('camera_2')

def process_frame(camera_id, frame, mode):
    start_time = datetime.now(timezone.utc).timestamp()
    
    # 전역 딕셔너리에서 현재 카메라의 상태와 카운트 정보를 가져옴
    stable_state = stable_state_dict.get(camera_id, {})
    change_count = change_count_dict.get(camera_id, {})

    # RT-DETR 입력 사이즈에 맞춰 리사이즈
    frame_for_model = cv2.resize(frame, (640, 640))

    # RT-DETR Inference (추론)
    results = model(frame_for_model, conf=0.1, verbose=False)

    # 차(car)가 감지된 영역을 255로 채운 마스크 생성
    occupied_mask = np.zeros((640, 640), dtype=np.uint8)
    
    if len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                if class_names[cls_idx] == 'car':
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(occupied_mask, (b[0], b[1]), (b[2], b[3]), 255, -1)

    # 주차 구역 점유 확인 및 지연 로직 적용
    for spot in parking_spots[camera_id]:
        spot_id = spot["id"]
        # 원본 좌표(1024)를 모델 입력 크기(640)로 스케일링
        x = int(spot["x"] * (640 / 1024))
        y = int(spot["y"] * (640 / 1024))

        if 0 <= x < 640 and 0 <= y < 640:
            # 1. 이번 프레임에서 감지된 즉각적인 상태 (Instant state)
            is_occupied_now = 1 if occupied_mask[y, x] > 0 else 0
            
            # 2. 현재 기록된 안정된 상태 (Last stable state)
            # 처음에 데이터가 없을 경우를 대비해 현재 상태로 초기화
            if spot_id not in stable_state:
                stable_state[spot_id] = is_occupied_now
                change_count[spot_id] = 0

            # 3. 상태 변화 감지 및 카운트 로직
            if is_occupied_now != stable_state[spot_id]:
                # 현재 감지된 상태가 기존 저장된 상태와 다르면 카운트 증가
                change_count[spot_id] = change_count.get(spot_id, 0) + 1
                
                # 설정한 MIN_CHANGE_FRAMES(8회) 이상 유지되었다면 상태 변경
                if change_count[spot_id] >= MIN_CHANGE_FRAMES:
                    stable_state[spot_id] = is_occupied_now
                    change_count[spot_id] = 0
            else:
                # 감지된 상태가 기존 상태와 같다면(안정적이라면) 카운트 초기화
                change_count[spot_id] = 0

            # 시각화 (최종 결정된 stable_state 기준)
            color = (0, 0, 255) if stable_state[spot_id] == 1 else (255, 0, 0)
            cv2.circle(frame_for_model, (x, y), 4, color, -1)
            cv2.putText(frame_for_model, f"{spot_id}", (x - 6, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 변경된 정보를 다시 전역 변수에 업데이트
    stable_state_dict[camera_id] = stable_state
    change_count_dict[camera_id] = change_count
    first_frame_processed_dict[camera_id] = True

    if mode != "server":
        cv2.imshow(f"Processed Frame ({camera_id})", frame_for_model)
        cv2.waitKey(1)

    # 결과 이미지 저장
    save_dir = os.path.join(OUTPUT_PATH, camera_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"frame_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(save_path, frame_for_model)


    end_time = datetime.now(timezone.utc).timestamp()
    processing_time = end_time - start_time

    return frame_for_model, processing_time

# ================== Server Mode ================== #
CSV_PATHS = {"camera_1": LATENCY_1, "camera_2": LATENCY_2}
for path in CSV_PATHS.values():
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_name", "upload_timestamp", "server_received_time", "processing_done_time", "upload_latency", "process_latency","total_latency"])

JETSON_CSV_PATHS = {"camera_1": JETSON_LATENCY_1, "camera_2": JETSON_LATENCY_2}
for path in JETSON_CSV_PATHS.values():
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['frame_name', "request_time", "upload_ready_time", "local_duration_sec"])
LAMBDA_CSV_PATHS = {"camera_1": LAMBDA_CSV_PATHS_CAM1, "camera_2": LAMBDA_CSV_PATHS_CAM2}
for path in LAMBDA_CSV_PATHS.values():
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_name", "lambda_network_delay", "processing_duration", "total_lambda_latency", "lambda_clock_offset"])

class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        if not content_type or not content_type.startswith('multipart/form-data'):
            self.send_response(400); self.end_headers(); return
        t2 = time.time()
    
        # 응답 기본값 (로직이 실패해도 최소한 이 정보는 나감)
        response_payload = {
            "status": "error", # 기본은 error로 설정
            "server_received_time": t2,
            "server_processing_done_time": t2  # 일단 T2와 같게 설정
        }
        try:
            fs = cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': content_type})

            file_item = fs['image']
            timestamp_str = fs.getvalue('timestamp')
            request_time_str = fs.getvalue('request_time')
            upload_time_str = fs.getvalue('upload_time')
            duration_str = fs.getvalue('duration')
            request_time = float(request_time_str)
            upload_time = float(upload_time_str)
            duration = float(duration_str)
            frame_name = fs.getvalue('frame_name')

            if file_item.file:
                img_bytes = file_item.file.read()
                np_img = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                if frame is not None:
                    camera_id = "camera_1" if self.path == "/upload/cam1" else "camera_2"

                    with open(JETSON_CSV_PATHS[camera_id], "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_name, request_time, upload_time, duration])
                    
                    upload_timestamp = float(timestamp_str) if timestamp_str else time.time()
                    server_received_time = time.time()

                    processing_info[camera_id]["start_time"] = upload_timestamp
                    processed_frame, processing_time = process_frame(camera_id, frame, "server")

                    processing_info[camera_id]["processing_time"] = processing_time
                    processing_info[camera_id]["server_sent_time"] = server_received_time
                    
                    processing_done_time = time.time()
                    processing_info[camera_id]["processing_done_time"] = processing_done_time
                    processing_info[camera_id]["frame_name"] = frame_name

                    with open(CSV_PATHS[camera_id], "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_name, upload_timestamp, server_received_time, processing_done_time,
                                        server_received_time - upload_timestamp,
                                        processing_done_time - server_received_time,
                                        processing_done_time - upload_timestamp])
                    
                    send_combined_to_lambda(processing_info, camera_id)

                    response_to_Jetson=time.time()
                    #cv2.imshow(f"Parking Detection (RT-DETR) - {camera_id}", processed_frame)
                    #cv2.waitKey(1)
            response_payload = {
                "status": "success",
                "server_received_time": server_received_time,
                "server_processing_done_time": response_to_Jetson
            }
        except Exception as e:
            print(f"Error processing request: {e}")
            t3 = time.time()
            response_payload = {
                "status": "error",
                "message": str(e),
                "server_received_time": t2,
                "server_processing_done_time": t3
            }
        finally:
            # 3. 로직 성공/실패 여부와 상관없이 응답 직전 시간(T3) 업데이트
            response_payload["server_processing_done_time"] = time.time()

            # 4. 최종 전송
            self.send_response(200) # 통신 자체는 성공했으므로 200
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

def run_server_mode():
    server = HTTPServer((HOST, PORT), ImageHandler)
    print(f"RT-DETR Server Running: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.server_close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "server"], default="server")
    args = parser.parse_args()
    run_server_mode()

if __name__ == "__main__":
    main()
