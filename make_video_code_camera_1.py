import cv2
import os
import glob
import re
import time

# 1. 경로 및 환경 설정
IMAGE_DIR = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/frame_check/camera_2"       # 분석 이미지가 생성되는 폴더 경로
OUTPUT_DIR = "/home/tuan-public/Desktop/yolo26_seg_NPU_improve/moo7/moo7_runtime/video_check/camera_2"      # 최종 동영상이 저장될 폴더 경로
BASE_VIDEO_NAME = "parking_check_camera_2_check" 
FPS = 1  # 동영상 파일의 기본 FPS 설정

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

file_idx=1
while True:
    VIDEO_NAME = f"{BASE_VIDEO_NAME}_{file_idx}.mp4"
    FINAL_VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_NAME)

    if not os.path.exists(FINAL_VIDEO_PATH):
        break
    
    file_idx += 1
print(f"save path -> {FINAL_VIDEO_PATH}")
# 파일 이름에서 숫자를 추출하는 함수
def extract_number(filepath):
    filename = os.path.basename(filepath)
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else 0

# 2. 초기 셋업: 이미 처리한 파일들을 추적하기 위한 셋(Set) 자료형
processed_files = set()
video_writer = None

# 프로그램 시작 시점에 이미 폴더에 있는 파일들은 완료된 것으로 간주하

current_files = glob.glob(os.path.join(IMAGE_DIR, "frame_*.jpg"))
processed_files.update(current_files)

cv2.namedWindow("Live Parking Monitor", cv2.WINDOW_NORMAL)

print("=" * 60)
print(" 실시간 모니터링 및 녹화를 시작합니다.")
print(" 새로운 이미지가 폴더에 추가되면 자동으로 화면에 띄우고 녹화합니다.")
print(" 종료하려면 이미지 창을 클릭한 후 'q'를 누르세요.")
print("=" * 60)

try:
    while True:
        # 3. 폴더 내 이미지 목록 수집 및 정렬
        search_path = os.path.join(IMAGE_DIR, "frame_*.jpg")
        all_files = glob.glob(search_path)
        all_files.sort(key=extract_number)
        
        # 4. 아직 처리(출력/녹화)되지 않은 새로운 파일들만 필터링
        new_files = [f for f in all_files if f not in processed_files]
        
        if new_files:
            # 새로 추가된 이미지들을 순서대로 처리
            for filepath in new_files:
                img = cv2.imread(filepath)
                
                if img is None:
                    # 이미지 생성 직후 시점이라 완벽히 쓰여지지 않아 읽기 실패한 경우, 잠시 후 재시도
                    continue
                
                # 5. 첫 이미지 유입 시 VideoWriter 초기화 (가로/세로 크기 동적 획득)
                if video_writer == None:
                    height, width, _ = img.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(FINAL_VIDEO_PATH, fourcc, FPS, (width, height))
                    print(f"[알림] 영상 녹화 시작 -> 해상도: {width}x{height}")
                
                # 화면 출력 및 비디오 파일에 프레임 추가
                cv2.imshow("Live Parking Monitor", img)
                video_writer.write(img)
                print(f"[재생/녹화 추가] {os.path.basename(filepath)}")
                
                # 읽기 완료 기록
                processed_files.add(filepath)
                
                # 이미지 간 최소한의 출력 보장을 위해 0.5초 대기하며 키 입력 체크
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
        else:
            # 6. [핵심] 새로운 이미지가 없을 때 대기 로직
            # 0.1초(100ms)마다 waitKey를 호출하여 윈도우 "응답 없음" 현상을 완벽히 방지합니다.
            if cv2.waitKey(100) & 0xFF == ord('q'):
                print("사용자에 의해 중지되었습니다.")
                break
                
except KeyboardInterrupt:
    print("\n사용자에 의해 모니터링이 강제 종료되었습니다.")

finally:
    # 7. 안전하게 자원 해제 및 비디오 최종 저장
    if video_writer is not None:
        video_writer.release()
        print(f"[저장 완료] 동영상이 안전하게 저장되었습니다: {FINAL_VIDEO_PATH}")
    else:
        print("[안내] 추가된 프레임이 없어 비디오가 생성되지 않았습니다.")
        
    cv2.destroyAllWindows()
