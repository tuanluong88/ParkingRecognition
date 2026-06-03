import os
import glob
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # YOLO 이미지 전처리 (기존과 동일)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # orig hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # wh padding
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = np.transpose(img, [2, 0, 1])
    return img, (h0, w0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Measure inference time of compiled MXQ model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the compiled MXQ model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory")
    
    args = parser.parse_args()

    # NPU 및 모델 초기화
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    # 이미지 경로 불러오기
    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    
    if not image_paths:
        print("지정된 디렉토리에서 이미지를 찾을 수 없습니다.")
        exit()

    print(f"총 {len(image_paths)}장 이미지에 대한 추론 속도 테스트를 시작합니다...\n")

    inference_times = []

    for i, img_path in enumerate(image_paths):
        # 1. 전처리 (시간 측정 제외)
        img, _ = preprocess_yolo(img_path)
        
        # 2. 추론 시작 시간 기록
        start_time = time.perf_counter()
        
        # 3. NPU 추론 실행
        outputs = model.infer([img])
        
        # 4. 추론 종료 시간 기록
        end_time = time.perf_counter()
        
        # 밀리초(ms) 단위로 변환
        infer_time_ms = (end_time - start_time) * 1000  
        inference_times.append(infer_time_ms)
        
        img_name = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_paths)}] {img_name} - 소요 시간: {infer_time_ms:.2f} ms")

    # 모델 메모리 해제
    model.dispose()

    # 통계 계산 및 출력
    print("\n" + "="*30)
    print("      속도 테스트 결과 요약      ")
    print("="*30)
    
    if len(inference_times) > 1:
        warmup_time = inference_times[0]
        # 첫 번째 웜업(Warm-up) 시간을 제외한 평균 계산
        avg_time = sum(inference_times[1:]) / (len(inference_times) - 1)
        
        print(f"첫 번째 이미지 (Warm-up) : {warmup_time:.2f} ms")
        print(f"평균 추론 시간           : {avg_time:.2f} ms (초기화 시간 제외)")
        print(f"예상 FPS (초당 프레임 수): {1000 / avg_time:.2f} FPS")
    else:
        avg_time = inference_times[0]
        print(f"추론 시간      : {avg_time:.2f} ms")
        print(f"예상 FPS       : {1000 / avg_time:.2f} FPS")
    print("="*30)