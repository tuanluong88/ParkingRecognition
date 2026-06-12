import os
import sys
import glob
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision  # NPU 경로와 동일한 batched_nms 사용을 위해 추가
from ultralytics import RTDETR


def check_gpu():
    """CUDA GPU 사용 가능 여부를 확인하고, 없으면 즉시 종료합니다."""
    if not torch.cuda.is_available():
        print("[오류] CUDA GPU를 사용할 수 없습니다.")
        print("  - GPU 드라이버 및 CUDA 설치 여부를 확인하세요.")
        print("  - GPU용 PyTorch가 설치되어 있는지 확인하세요. (torch.cuda.is_available() = False)")
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[확인] GPU 감지됨: {gpu_name}")
    print(f"[확인] 사용 가능한 GPU 수: {torch.cuda.device_count()}")


def preprocess_yolo(img_path: str, img_size=(640, 640)):
    """
    기존 evaluate_mxq_seg_labels.py의 전처리 방식과 동일하게 letterbox 적용.
    (ultralytics가 내부적으로 동일한 처리를 수행하므로,
     원본 이미지 크기(h0, w0)만 반환하면 됩니다.)
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")
    h0, w0 = img.shape[:2]
    return h0, w0


def compute_iou(box1, box2):
    """두 바운딩 박스(x1, y1, x2, y2)의 IoU를 계산합니다."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    u_area = area1 + area2 - i_area

    return i_area / u_area if u_area > 0 else 0


def load_yolo_seg_labels(label_path, img_w, img_h):
    """
    YOLO segmentation 포맷 라벨(class_id x1 y1 x2 y2 ...)을 읽어
    절대 픽셀 좌표의 바운딩 박스로 변환합니다.
    (기존 코드와 동일한 로직)
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # 유효한 segmentation 라벨: class_id + 최소 6개 좌표(삼각형)
            if len(parts) >= 7:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # x, y 좌표 분리 (교대로 배치)
                x_coords = coords[0::2]
                y_coords = coords[1::2]

                # 폴리곤을 감싸는 최소 직사각형 계산
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)

                # 정규화 좌표(0.0 ~ 1.0) → 절대 픽셀 좌표 변환
                x1 = x_min * img_w
                y1 = y_min * img_h
                x2 = x_max * img_w
                y2 = y_max * img_h

                gt_boxes.append({
                    'class_id': class_id,
                    'box': [x1, y1, x2, y2],
                    'matched': False
                })
    return gt_boxes


def evaluate_predictions(predictions, gt_boxes, iou_threshold=0.5):
    """
    단일 이미지에 대한 TP, FP, FN을 계산합니다.
    (기존 코드와 동일한 로직)
    """
    TP, FP = 0, 0

    # confidence 내림차순 정렬
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)

    for pred in predictions:
        pred_box = pred[:4]   # [x1, y1, x2, y2]
        pred_cls = int(pred[5])

        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt['class_id'] == pred_cls and not gt['matched']:
                iou = compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

        if best_iou >= iou_threshold:
            gt_boxes[best_gt_idx]['matched'] = True
            TP += 1
        else:
            FP += 1

    # 매칭되지 않은 GT = False Negative
    FN = sum(1 for gt in gt_boxes if not gt['matched'])

    return TP, FP, FN


def parse_rtdetr_results(results, conf_thres):
    """
    ultralytics RT-DETR 추론 결과(Results 객체)를
    기존 코드와 동일한 형식인 [[x1, y1, x2, y2, conf, cls], ...] 로 변환합니다.

    ultralytics Results.boxes 속성:
      - xyxy  : (N, 4) 절대 픽셀 좌표 [x1, y1, x2, y2]
      - conf  : (N,)   confidence score
      - cls   : (N,)   class index
    """
    image_preds = []

    if results is None or results.boxes is None:
        return image_preds

    boxes = results.boxes

    # 텐서 → NumPy 변환
    xyxy = boxes.xyxy.cpu().numpy()   # (N, 4)
    confs = boxes.conf.cpu().numpy()  # (N,)
    clses = boxes.cls.cpu().numpy()   # (N,)

    for i in range(len(xyxy)):
        conf = float(confs[i])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = xyxy[i]
        cls = float(clses[i])
        image_preds.append([float(x1), float(y1), float(x2), float(y2), conf, cls])

    return image_preds


def apply_batched_nms(image_preds, iou_threshold=0.7):
    """
    NPU 후처리(RTDETRPostProcess)와 *완전히 동일한* batched_nms를
    GPU 예측에도 적용하여 양쪽 후처리 스택을 동등하게 맞춥니다.

    - RT-DETR은 NMS-free 설계라 FP32(GPU) 모델은 중복 박스가 거의 없습니다.
      따라서 이 NMS를 통과시켜도 GPU F1은 거의 변하지 않으며,
      이는 "양자화된 NPU 모델에만 NMS를 적용한 것이 아니라,
      양쪽에 동일한 표준 후처리를 적용한 공정 비교"임을 보장합니다.
    - image_preds: [[x1, y1, x2, y2, conf, cls], ...]
    """
    if len(image_preds) == 0:
        return image_preds

    preds = torch.tensor(image_preds, dtype=torch.float32)
    boxes = preds[:, :4]
    scores = preds[:, 4]
    labels = preds[:, 5]

    keep = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold)
    return preds[keep].tolist()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="RT-DETR (.pt) 모델의 성능을 segmentation 라벨 기준으로 평가합니다."
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="RT-DETR .pt 모델 경로"
    )
    parser.add_argument(
        "--dataset-dir", type=str, required=True,
        help="데이터셋 디렉토리 경로 ('images' 및 'labels' 하위 폴더 필요)"
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="추론 시 사용할 이미지 크기 (기본값: 640)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.35,
        help="예측 결과 confidence 임계값 (기본값: 0.35)"
    )
    parser.add_argument(
        "--nms-iou-thres", type=float, default=0.45,
        help="ultralytics 내부 NMS IoU 임계값 (기본값: 0.45)"
    )
    parser.add_argument(
        "--eval-iou-thres", type=float, default=0.45,
        help="TP 판정을 위한 IoU 임계값 (기본값: 0.45)"
    )
    parser.add_argument(
        "--post-nms-iou", type=float, default=0.7,
        help="NPU 경로와 동일하게 적용하는 후처리 batched_nms IoU 임계값 (기본값: 0.7). "
             "공정 비교를 위해 NPU 측 RTDETRPostProcess와 동일한 값을 사용하세요."
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="추론 장치 (예: '0', '1', '0,1'). 기본값: 0 (GPU 강제 사용)"
    )

    args = parser.parse_args()

    # ── GPU 강제 확인 (CPU 실행 차단) ─────────────────────────────────────
    check_gpu()

    # ── 모델 로드 ──────────────────────────────────────────────────────────
    print(f"모델 로드 중: {args.model_path}")
    model = RTDETR(args.model_path)

    # ── 데이터셋 경로 설정 ──────────────────────────────────────────────────
    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    # 이미지 파일만 필터링 (라벨 .txt 등 제외)
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = [
        p for p in image_paths
        if os.path.splitext(p)[1].lower() in valid_extensions
    ]
    label_dir = os.path.join(args.dataset_dir, "labels")

    total_TP = 0
    total_FP = 0
    total_FN = 0

    print(f"평가 시작: 총 {len(image_paths)}개 이미지")
    print(f"  conf_thres    = {args.conf_thres}")
    print(f"  nms_iou_thres = {args.nms_iou_thres}")
    print(f"  eval_iou_thres= {args.eval_iou_thres}")
    print(f"  post_nms_iou  = {args.post_nms_iou}  (NPU 경로와 동일한 후처리 NMS)")
    print("-" * 50)

    for img_path in image_paths:
        # 1. 원본 이미지 크기 획득 (라벨 좌표 역변환에 필요)
        h0, w0 = preprocess_yolo(img_path)

        # 2. RT-DETR 추론
        #    ultralytics는 내부적으로 letterbox 전처리 + NMS 후처리까지 수행하며,
        #    결과 박스는 원본 이미지 해상도 기준 절대 픽셀 좌표로 반환됩니다.
        results_list = model.predict(
            source=img_path,
            imgsz=args.img_size,
            conf=args.conf_thres,
            iou=args.nms_iou_thres,
            device=args.device,
            verbose=False,
        )

        # 3. 첫 번째(단일) 이미지 결과 파싱
        #    model.predict()는 리스트를 반환하므로 [0] 인덱스 사용
        result = results_list[0] if results_list else None
        image_preds = parse_rtdetr_results(result, conf_thres=args.conf_thres)

        # 3-1. ⭐ NPU 경로와 동일한 후처리 NMS 적용 (공정 비교)
        image_preds = apply_batched_nms(image_preds, iou_threshold=args.post_nms_iou)

        # 4. GT 라벨 로드 (segmentation 폴리곤 → bounding box 변환)
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)

        gt_boxes = load_yolo_seg_labels(label_path, 640, 640)

        # 5. TP / FP / FN 계산 (기존 코드와 동일한 로직)
        TP, FP, FN = evaluate_predictions(image_preds, gt_boxes, args.eval_iou_thres)

        total_TP += TP
        total_FP += FP
        total_FN += FN

    # ── 최종 성능 지표 출력 ────────────────────────────────────────────────
    precision = (
        total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    )
    recall = (
        total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    print("\n--- 성능 평가 결과 ---")
    print(f"Total True Positives  (TP): {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print("---------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1_score:.4f}")