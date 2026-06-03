import os
import glob
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

# 직접 완성하신 하이브리드 모듈 임포트
from inference_mxq import HybridRTDETR
from postprocess import RTDETRPostProcess

def preprocess_yolo_speed(img_path: str, img_size=(640, 640)):
    """NPU 모델이 기대하는 CHW 구조 및 uint8 형태를 유지하는 전처리 함수"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    r = min(img_size[0] / h0, img_size[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    img = np.transpose(img, [2, 0, 1])
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, axis=0)
    
    return img

if __name__ == "__main__":
    parser = ArgumentParser(description="Measure End-to-End inference time of Hybrid RT-DETR")
    parser.add_argument("--mxq-path", type=str, required=True, help="Path to the compiled MXQ model")
    parser.add_argument("--pt-path", type=str, required=True, help="Path to the PyTorch decoder model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()

    print("시스템을 초기화합니다...")
    # 1. 하이브리드 모델 & 후처리 모듈 초기화
    hybrid_model = HybridRTDETR(args.mxq_path, args.pt_path)
    postprocess = RTDETRPostProcess(args.conf_thres)

    # 2. 이미지 경로 불러오기
    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    
    if not image_paths:
        print("지정된 디렉토리에서 이미지를 찾을 수 없습니다.")
        exit()

    print(f"총 {len(image_paths)}장 이미지에 대한 End-to-End 속도 테스트를 시작합니다...\n")

    total_times = []
    pre_times = []
    infer_times = []
    post_times = []

    # 성능 측정 루프
    for i, img_path in enumerate(image_paths):
        # --- 전체 사이클 시작 ---
        t_cycle_start = time.perf_counter()
        
        # [Step 1] 전처리 (Pre-processing)
        t0 = time.perf_counter()
        img = preprocess_yolo_speed(img_path)
        t1 = time.perf_counter()
        
        # [Step 2] 모델 추론 (Inference: NPU 인코더 + PyTorch 디코더)
        outputs = hybrid_model(img)
        t2 = time.perf_counter()
        
        # [Step 3] 후처리 (Post-processing)
        if isinstance(outputs, tuple):
            outputs_list = list(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs_list = [outputs]
        else:
            outputs_list = outputs
            
        result = postprocess(outputs_list)
        t3 = time.perf_counter()
        # --- 전체 사이클 종료 ---

        # 밀리초(ms) 단위 계산
        pre_ms = (t1 - t0) * 1000
        infer_ms = (t2 - t1) * 1000
        post_ms = (t3 - t2) * 1000
        total_ms = (t3 - t_cycle_start) * 1000
        
        pre_times.append(pre_ms)
        infer_times.append(infer_ms)
        post_times.append(post_ms)
        total_times.append(total_ms)
        
        img_name = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_paths)}] {img_name} | Total: {total_ms:.2f}ms (Pre: {pre_ms:.2f}ms, Infer: {infer_ms:.2f}ms, Post: {post_ms:.2f}ms)")

    # 메모리 해제
    hybrid_model.dispose()

    # 통계 계산 및 출력 (첫 번째 이미지는 하드웨어 워밍업으로 간주하여 제외)
    print("\n" + "="*55)
    print("      End-to-End 속도 테스트 결과 요약 (Warm-up 제외)      ")
    print("="*55)
    
    if len(total_times) > 1:
        avg_total = sum(total_times[1:]) / (len(total_times) - 1)
        avg_pre = sum(pre_times[1:]) / (len(pre_times) - 1)
        avg_infer = sum(infer_times[1:]) / (len(infer_times) - 1)
        avg_post = sum(post_times[1:]) / (len(post_times) - 1)
        
        print(f"평균 전체 소요 시간 (Total) : {avg_total:.2f} ms")
        print(f"  ├─ 평균 전처리 (Pre)      : {avg_pre:.2f} ms")
        print(f"  ├─ 평균 하이브리드 추론   : {avg_infer:.2f} ms (NPU + CPU)")
        print(f"  └─ 평균 후처리 (Post)     : {avg_post:.2f} ms")
        print("-" * 55)
        print(f"최종 예상 FPS (End-to-End)  : {1000 / avg_total:.2f} FPS")
    else:
        print("이미지가 부족하여 평균을 계산할 수 없습니다.")
    print("="*55)