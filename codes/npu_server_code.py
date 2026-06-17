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
from collections import OrderedDict  # 완벽한 순서 제어를 위해 유지
from npu_yolo_function import NPUYoloModel

# ================== Setup and Paths ================== #
#lambda_url 경로
LAMBDA_URL = "https://eqyvtgycwxeirvve7cldswlk6a0eungg.lambda-url.ap-northeast-2.on.aws/update-button-status"
MIN_CHANGE_FRAMES = 8      
SEND_INTERVAL = 0       
# mxq 모델 경로
NPU_MODEL_PATH = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/simplified_skku_yolo26_epoch1000_seg.mxq"
# 이미지 저장 경로
OUTPUT_PATH = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/frame_check"
# 주차장 점유 확인 점 위치 json 파일 경로
JSON_PATH_ANGLE_1 = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/dot/json_final/dot/parking_spots_zoneA_seg_dot.json"
JSON_PATH_ANGLE_2 = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/dot/json_final/dot/parking_spots_zoneB_seg_dot.json"
#latency csv 파일 경로
JETSON_LATENCY_1='/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/Jetson_latency_cam1.csv'
JETSON_LATENCY_2='/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/Jetson_latency_cam2.csv'
LATENCY_1="/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/latency_cam1.csv"
LATENCY_2="/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/latency_cam2.csv"
LAMBDA_CSV_PATHS_CAM1="/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/latency_lambda1.csv"
LAMBDA_CSV_PATHS_CAM2="/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/latency/latency_lambda2.csv"
#check results
STATUS_LOG_CSV ="/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/results_check/results_check.csv"
HOST = "0.0.0.0"
PORT = 8080

last_combined_send_time = time.time()
last_sent_time_dict = {"camera_1": 0, "camera_2": 0}

with open(JSON_PATH_ANGLE_1) as f:
    parking_spots_angle_1 = json.load(f)
with open(JSON_PATH_ANGLE_2) as f:
    parking_spots_angle_2 = json.load(f)

# 전역 변수 구조 설정
stable_state_dict = {
    "camera_1": OrderedDict(),
    "camera_2": OrderedDict()
}
change_count_dict = {
    "camera_1": OrderedDict(),
    "camera_2": OrderedDict()
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

#load NPU model
model = NPUYoloModel(model_path=NPU_MODEL_PATH, conf_thres=0.1)

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
    
    # Lambda로 보낼 배열을 뽑기 전에 문자열 ID를 진짜 순서로 정렬
    # 10번대 데이터가 사전순으로 새치기하는 현상 방지
    try:
        sorted_spots = sorted(spots, key=lambda x: int(x["id"]))
    except ValueError:
        sorted_spots = sorted(spots, key=lambda x: str(x["id"]))
        
    state_array = [state_dict.get(spot["id"], 0) for spot in sorted_spots]
    return state_array

def send_combined_to_lambda(processing_info, camera_id):
    global last_combined_send_time
    current_time = time.time()
    for cam in ["camera_1", "camera_2"]:
        processing_info[cam]["lambda_send_time"] = current_time

    cam1_status = convert_stable_state_to_array("camera_1")
    cam2_status = convert_stable_state_to_array("camera_2")

    combined_payload = {
        "button_status": {
            "camera_1": cam1_status,
            "camera_2": cam2_status
        },
        "latency_info": {
            "camera_1": processing_info["camera_1"],
            "camera_2": processing_info["camera_2"]
        }
    }
    combined_payload["send_camera_id"] = camera_id

    current_frame_name = processing_info[camera_id].get("frame_name", "unknown")
    
    try:
        with open(STATUS_LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                current_frame_name, 
                camera_id, 
                json.dumps(cam1_status), 
                json.dumps(cam2_status)
            ])
        print(f"[Log] Saved status for {current_frame_name} to CSV.")
    except Exception as log_error:
        print(f"[Log] CSV Save error: {log_error}")

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
            
            lambda_network_delay = ((v4 - v1) - (v3 - v2)) / 2
            lambda_clock_offset = ((v2 - v1) + (v3 - v4)) / 2
            total_lambda_latency = lambda_network_delay + processing_duration
            
            with open(LAMBDA_CSV_PATHS[camera_id], "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([frame_name, lambda_network_delay, processing_duration, total_lambda_latency, lambda_clock_offset])
    except Exception as e:
        print(f"[Lambda] Send error: {e}")
    last_sent_time_dict["camera_1"] = current_time
    last_sent_time_dict["camera_2"] = current_time

def initialize_camera(camera_id):
    spots = parking_spots[camera_id]
    
    stable_state_dict[camera_id] = OrderedDict()
    change_count_dict[camera_id] = OrderedDict()
    
    # 초기화 시점부터 JSON에 적힌 문자를 진짜 숫자 기준으로 오름차순 정렬
    # 1, 2, 3 ... 9, 10, 11 순서대로 방이 먼저 개설되어 순서 꼬임을 방지
    try:
        sorted_spots = sorted(spots, key=lambda x: int(x["id"]))
    except ValueError:
        sorted_spots = sorted(spots, key=lambda x: str(x["id"]))
    
    for spot in sorted_spots:
        spot_id = spot["id"]
        stable_state_dict[camera_id][spot_id] = 0
        change_count_dict[camera_id][spot_id] = 0
        
    first_frame_processed_dict[camera_id] = False

# 초기화 수행
initialize_camera('camera_1')
initialize_camera('camera_2')

def process_frame(camera_id, frame, mode, frame_name=None):
    start_time = datetime.now(timezone.utc).timestamp()
    
    stable_state = stable_state_dict.get(camera_id)
    change_count = change_count_dict.get(camera_id)

    frame_for_model = cv2.resize(frame, (640, 640))
    results = model.predict(frame_for_model)

    occupied_mask = np.zeros((640, 640), dtype=np.uint8)
    r = results[0]
    if r.masks is not None:
        for i in range(len(r.masks)):
            if int(r.boxes[i, 5]) == 0:
                mask_data = (r.masks[i] * 255).astype(np.uint8)
                occupied_mask = cv2.bitwise_or(occupied_mask, mask_data)

    # 주차 구역 점유 확인 및 8초 지연 로직 루프 (항상 원본 JSON 리스트 순서대로 순회)
    for spot in parking_spots[camera_id]:
        spot_id = spot["id"]
        x = int(spot["x"] * (640 / 1024))
        y = int(spot["y"] * (640 / 1024))

        if 0 <= x < 640 and 0 <= y < 640:
            is_occupied_now = 1 if occupied_mask[y, x] > 0 else 0
            
            # 고유 key인 spot_id로 매칭
            last_stable = stable_state[spot_id]

            if is_occupied_now != last_stable:
                change_count[spot_id] += 1
                if change_count[spot_id] >= MIN_CHANGE_FRAMES:
                    stable_state[spot_id] = is_occupied_now
                    change_count[spot_id] = 0
            else:
                change_count[spot_id] = 0

            # 시각화
            color = (0, 0, 255) if stable_state[spot_id] == 1 else (255, 0, 0)
            cv2.circle(frame_for_model, (x, y), 2, color, -1)
            cv2.putText(frame_for_model, f"{spot_id}", (x - 4, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1)

    # 가공 및 필터링 완료된 상태를 전역 변수에 안전하게 대입
    stable_state_dict[camera_id] = stable_state
    change_count_dict[camera_id] = change_count
    first_frame_processed_dict[camera_id] = True

    if mode != "server":
        cv2.imshow(f"Processed Frame ({camera_id})", frame_for_model)
        cv2.waitKey(1)
    
    #이미지를 회전해서 저장할 시 활성화
    #frame_for_model = cv2.rotate(frame_for_model, cv2.ROTATE_180)


    save_dir = os.path.join(OUTPUT_PATH, camera_id)
    os.makedirs(save_dir, exist_ok=True)

    if frame_name:
        filename = os.path.splitext(frame_name)[0] 
        save_path = os.path.join(save_dir, f"{filename}.jpg")
    else:
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

if not os.path.exists(STATUS_LOG_CSV):
    with open(STATUS_LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_name", "camera_id", "camera_1_status", "camera_2_status"])

class ImageHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        if not content_type or not content_type.startswith('multipart/form-data'):
            self.send_response(400); self.end_headers(); return
        t2 = time.time()
    
        response_payload = {
            "status": "error",
            "server_received_time": t2,
            "server_processing_done_time": t2  
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
                    processed_frame, processing_time = process_frame(camera_id, frame, "server", frame_name)

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

                    response_to_Jetson = time.time()
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
            response_payload["server_processing_done_time"] = time.time()
            response_payload["status"] = "success" if "processed_frame" in locals() else "error"

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

def run_server_mode():
    server = HTTPServer((HOST, PORT), ImageHandler)
    print(f"YOLO26_seg_NPU server Running: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        model.close()
        server.server_close()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "server"], default="server")
    args = parser.parse_args()
    run_server_mode()

if __name__ == "__main__":
    main()