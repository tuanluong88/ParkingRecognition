import os
import glob
from argparse import ArgumentParser
import torch
import cv2
import numpy as np

# 기존에 성공적으로 완성하신 하이브리드 클래스와 후처리 모듈을 불러옵니다.
from inference_mxq import HybridRTDETR
from postprocess import RTDETRPostProcess

def preprocess_yolo_eval(img_path: str, img_size=(640, 640)):
    """
    바운딩 박스 평가를 위해 원본 이미지의 크기(h0, w0)를 함께 반환하도록
    수정된 전처리 함수입니다.
    """
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
    
    return img, (h0, w0)

def compute_iou(box1, box2):
    """두 바운딩 박스의 IoU(Intersection over Union)를 계산합니다."""
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
    YOLO 세그멘테이션 포맷 라벨을 읽어와서 절대 픽셀 좌표의 바운딩 박스로 변환합니다.
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 7:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                x1 = x_min * img_w
                y1 = y_min * img_h
                x2 = x_max * img_w
                y2 = y_max * img_h
                
                gt_boxes.append({'class_id': class_id, 'box': [x1, y1, x2, y2], 'matched': False})
    return gt_boxes

def evaluate_predictions(predictions, gt_boxes, iou_threshold=0.5):
    """단일 이미지에 대한 TP(True Positives), FP(False Positives), FN(False Negatives)를 계산합니다."""
    TP, FP = 0, 0
    
    # 신뢰도(Confidence) 기준으로 내림차순 정렬
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    
    for pred in predictions:
        pred_box = pred[:4]
        pred_conf = pred[4]
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
            
    FN = sum(1 for gt in gt_boxes if not gt['matched'])
    
    return TP, FP, FN

if __name__ == "__main__":
    parser = ArgumentParser(description="하이브리드 RT-DETR 성능 평가 스크립트")
    parser.add_argument("--mxq-path", type=str, required=True, help="컴파일된 NPU MXQ 모델 경로")
    parser.add_argument("--pt-path", type=str, required=True, help="PyTorch 디코더 모델 경로")
    parser.add_argument("--dataset-dir", type=str, required=True, help="데이터셋 폴더 경로 (images, labels 폴더 포함)")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="예측 신뢰도 임계값")
    parser.add_argument("--eval-iou-thres", type=float, default=0.50, help="정답(TP) 인정을 위한 IoU 임계값")
    
    args = parser.parse_args()

    # 1. 하이브리드 시스템 초기화
    hybrid_model = HybridRTDETR(args.mxq_path, args.pt_path)
    postprocess = RTDETRPostProcess(args.conf_thres)

    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    label_dir = os.path.join(args.dataset_dir, "labels")

    total_TP = 0
    total_FP = 0
    total_FN = 0

    print(f"총 {len(image_paths)}개의 이미지에 대한 평가를 시작합니다...")

    for img_path in image_paths:
        img, (h0, w0) = preprocess_yolo_eval(img_path)
        
        # 2. 하이브리드 추론 (NPU -> 파이토치)
        outputs = hybrid_model(img)
        
        if isinstance(outputs, tuple):
            outputs_list = list(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs_list = [outputs]
        else:
            outputs_list = outputs
            
        result = postprocess(outputs_list)
        
        # 3. 결과 파싱 및 라벨 로드
        image_preds = result[0] if result is not None and len(result) > 0 else []
        if hasattr(image_preds, 'cpu'):
            image_preds = image_preds.cpu().numpy()
            
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        
        gt_boxes = load_yolo_seg_labels(label_path, w0, h0)
        
        # 4. 성능 평가 (TP, FP, FN 집계)
        TP, FP, FN = evaluate_predictions(image_preds, gt_boxes, args.eval_iou_thres)
        
        total_TP += TP
        total_FP += FP
        total_FN += FN

    hybrid_model.dispose()

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- 성능 평가 결과 (Performance Results) ---")
    print(f"Total True Positives (TP) : {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print("------------------------------------------")
    print(f"정밀도 (Precision) : {precision:.4f}")
    print(f"재현율 (Recall)    : {recall:.4f}")
    print(f"F1 점수 (F1 Score) : {f1_score:.4f}")