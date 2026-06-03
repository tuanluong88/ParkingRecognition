import os
import glob
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPostProcessAnchorless

# ... [preprocess_yolo, compute_iou, load_yolo_labels 함수는 기존과 동일] ...

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/augment.py#L1535
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  # orig hw
    r = min(img_size[0] / h0, img_size[1] / w0)  # ratio
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2  # to center the image
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
    img = np.transpose(img, [2, 0, 1])
    return img, (h0, w0)

def compute_iou(box1, box2):
    """Compute Intersection over Union (IoU) of two bounding boxes (x1, y1, x2, y2)."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    
    i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    u_area = area1 + area2 - i_area
    
    return i_area / u_area if u_area > 0 else 0

def load_yolo_labels(label_path, img_w, img_h):
    """Loads YOLO format labels and converts them to absolute coordinates."""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # Convert normalized xywh to absolute x1, y1, x2, y2
                x1 = (x_center - w / 2) * img_w
                y1 = (y_center - h / 2) * img_h
                x2 = (x_center + w / 2) * img_w
                y2 = (y_center + h / 2) * img_h
                
                gt_boxes.append({'class_id': class_id, 'box': [x1, y1, x2, y2], 'matched': False})
    return gt_boxes

def evaluate_predictions_by_class(predictions, gt_boxes, iou_threshold=0.5):
    """클래스별로 TP, FP, FN을 계산하여 반환합니다."""
    # 결과를 담을 딕셔너리 (key: class_id, value: [TP, FP, FN])
    class_metrics = defaultdict(lambda: [0, 0, 0])
    
    # 예측값들을 자신감 순으로 정렬
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    # 해당 이미지에 존재하는 모든 GT 클래스 파악
    all_classes = set([gt['class_id'] for gt in gt_boxes] + [int(p[5]) for p in predictions])
    
    for pred in predictions:
        pred_box = pred[:4]
        pred_cls = int(pred[5])
        
        best_iou = 0
        best_gt_idx = -1
        
        # 동일 클래스 내에서 가장 잘 맞는 GT 찾기
        for i, gt in enumerate(gt_boxes):
            if gt['class_id'] == pred_cls and not gt['matched']:
                iou = compute_iou(pred_box, gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    
        if best_iou >= iou_threshold:
            gt_boxes[best_gt_idx]['matched'] = True
            class_metrics[pred_cls][0] += 1  # TP 증가
        else:
            class_metrics[pred_cls][1] += 1  # FP 증가
            
    # 매칭되지 않은 GT들을 FN으로 집계[cite: 2]
    for gt in gt_boxes:
        if not gt['matched']:
            class_metrics[gt['class_id']][2] += 1  # FN 증가
            
    return class_metrics

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate MXQ model performance on a dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the compiled MXQ model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--nms-iou-thres", type=float, default=0.45)
    parser.add_argument("--eval-iou-thres", type=float, default=0.50)
    
    args = parser.parse_args()

    # Setup qbruntime 및 모델 로드 로직 동일
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPostProcessAnchorless(args.conf_thres, args.nms_iou_thres)

    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    label_dir = os.path.join(args.dataset_dir, "labels")

    # 모든 이미지에 대한 클래스별 누적 성능 (class_id: [total_TP, total_FP, total_FN])
    global_class_metrics = defaultdict(lambda: [0, 0, 0])

    print(f"Starting evaluation on {len(image_paths)} images...")

    for img_path in image_paths:
        img, (h0, w0) = preprocess_yolo(img_path)
        outputs = model.infer([img])
        raw_result = postprocess(outputs)
        preds = raw_result[0].tolist() if raw_result is not None else []
        
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(label_dir, os.path.splitext(img_filename)[0] + ".txt")
        gt_boxes = load_yolo_labels(label_path, w0, h0)
        
        # 이미지별 클래스 지표 계산
        img_metrics = evaluate_predictions_by_class(preds, gt_boxes, args.eval_iou_thres)
        
        # 전체 누적
        for cls, counts in img_metrics.items():
            global_class_metrics[cls][0] += counts[0]
            global_class_metrics[cls][1] += counts[1]
            global_class_metrics[cls][2] += counts[2]

    model.dispose()

    # --- YOLO 스타일 평균(Macro Average) 계산 ---
    precisions, recalls, f1s = [], [], []
    
    print("\n--- Class-wise Performance ---")
    for cls in sorted(global_class_metrics.keys()):
        tp, fp, fn = global_class_metrics[cls]
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        
        print(f"Class {cls} - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f} (TP:{tp}, FP:{fp}, FN:{fn})")

    # 모든 클래스의 지표를 더한 뒤 클래스 수로 나눔 (Macro Average)
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1s) if f1s else 0

    print("\n--- Final Average Results (Macro) ---")
    print(f"Mean Precision : {avg_precision:.4f}")
    print(f"Mean Recall    : {avg_recall:.4f}")
    print(f"Mean F1 Score  : {avg_f1:.4f}")