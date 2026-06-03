import os
import glob
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPostProcessAnchorless

def preprocess_yolo(img_path: str, img_size=(640, 640)):
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
    parser = ArgumentParser(description="Measure End-to-End inference time of MXQ model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the compiled MXQ model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    
    args = parser.parse_args()

    # NPU 및 모델 초기화
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    # 후처리 모듈 초기화
    postprocess = YoloPostProcessAnchorless(args.conf_thres, args.nms_iou_thres)

    # 이미지 경로 불러오기
    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    
    if not image_paths:
        print("지정된 디렉토리에서 이미지를 찾을 수 없습니다.")
        exit()

    print(f"총 {len(image_paths)}장 이미지에 대한 End-to-End 속도 테스트를 시작합니다...\n")

    total_times = []
    pre_times = []
    infer_times = []
    post_times = []

    for i, img_path in enumerate(image_paths):
        # --- 전체 사이클 시작 ---
        t_cycle_start = time.perf_counter()
        
        # 1. 전처리 (Pre-processing)
        t0 = time.perf_counter()
        img, _ = preprocess_yolo(img_path)
        t1 = time.perf_counter()
        
        # 2. 모델 추론 (Inference)
        outputs = model.infer([img])
        t2 = time.perf_counter()
        
        # 3. 후처리 (Post-processing)
        result = postprocess(outputs)
        # 스칼라 에러 방지를 위한 텐서 변환 포함
        image_preds = result[0] if len(result) > 0 else []
        if hasattr(image_preds, 'cpu'):
            image_preds = image_preds.cpu().numpy()
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

    # 모델 메모리 해제
    model.dispose()

    # 통계 계산 및 출력 (첫 번째 Warm-up 제외)
    print("\n" + "="*50)
    print("      End-to-End 속도 테스트 결과 요약 (Warm-up 제외)      ")
    print("="*50)
    
    if len(total_times) > 1:
        avg_total = sum(total_times[1:]) / (len(total_times) - 1)
        avg_pre = sum(pre_times[1:]) / (len(pre_times) - 1)
        avg_infer = sum(infer_times[1:]) / (len(infer_times) - 1)
        avg_post = sum(post_times[1:]) / (len(post_times) - 1)
        
        print(f"평균 전체 소요 시간 (Total) : {avg_total:.2f} ms")
        print(f"  ├─ 평균 전처리 (Pre)      : {avg_pre:.2f} ms")
        print(f"  ├─ 평균 NPU 추론 (Infer)  : {avg_infer:.2f} ms")
        print(f"  └─ 평균 후처리 (Post)     : {avg_post:.2f} ms")
        print("-" * 50)
        print(f"최종 예상 FPS (End-to-End)  : {1000 / avg_total:.2f} FPS")
    else:
        print("이미지가 부족하여 평균을 계산할 수 없습니다.")
    print("="*50)