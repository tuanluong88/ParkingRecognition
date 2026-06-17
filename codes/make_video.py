import cv2
import os
import glob
import re

# 1. 경로 설정
IMAGE_DIR = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/frame_check/camera_1"
OUTPUT_DIR = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/video_check/camera_1"
BASE_VIDEO_NAME = "parking_check_camera_1" 
FPS = 1

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 파일명에서 마지막 숫자만 추출하는 함수 (예: cam1_10.jpg -> 10)
def extract_number(filepath):
    filename = os.path.basename(filepath)
    # 파일명 끝부분의 숫자 패턴을 찾음
    numbers = re.findall(r'_(\d+)\.(?:jpg|jpeg|png)$', filename)
    return int(numbers[0]) if numbers else 0

# 자동 파일명 생성 로직
file_idx = 1
while True:
    VIDEO_NAME = f"{BASE_VIDEO_NAME}_{file_idx}.mp4"
    FINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_NAME)
    if not os.path.exists(FINAL_VIDEO_PATH):
        break
    file_idx += 1

print(f"저장 경로: {FINAL_VIDEO_PATH}")

# 2. 초기 셋업
processed_files = set()
video_writer = None

# 프로그램 시작 시 이미 존재하는 파일들은 완료 처리
current_files = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
processed_files.update(current_files)

cv2.namedWindow("Live Monitor", cv2.WINDOW_NORMAL)

print("=" * 60)
print(" 실시간 모니터링 시작 (카메라별 독립 실행)")
print(" 종료하려면 창을 클릭하고 'q'를 누르세요.")
print("=" * 60)

try:
    while True:
        # 3. 모든 이미지 파일 검색
        all_files = glob.glob(os.path.join(IMAGE_DIR, "*.*"))
        all_files.sort(key=extract_number)
        
        # 4. 새 파일 필터링
        new_files = [f for f in all_files if f not in processed_files]
        
        if new_files:
            for filepath in new_files:
                img = cv2.imread(filepath)
                if img is None: continue
                
                # 비디오 라이터 초기화
                if video_writer is None:
                    height, width, _ = img.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(FINAL_VIDEO_PATH, fourcc, FPS, (width, height))
                    print(f"[알림] 녹화 시작: {width}x{height}")
                
                cv2.imshow("Live Monitor", img)
                video_writer.write(img)
                print(f"[처리 완료] {os.path.basename(filepath)}")
                
                processed_files.add(filepath)
                
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
        else:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
                
except KeyboardInterrupt:
    print("\n중지됨.")

finally:
    if video_writer is not None:
        video_writer.release()
        print(f"[저장 완료] {FINAL_VIDEO_PATH}")
    cv2.destroyAllWindows()