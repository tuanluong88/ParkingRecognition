import os
import glob
import argparse
import cv2
import numpy as np
import qbruntime

# 기존에 작성하신 후처리 모듈 임포트
from postprocess import Yolo26SegPostProcessE2E

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    """추론용 이미지 전처리 (Uint8 최적화 유지)"""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    
    r = min(img_size[0] / h0, img_size[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]
    top, bottom = int(round(dh // 2)), int(round(dh - dh // 2))
    left, right = int(round(dw // 2)), int(round(dw - dw // 2))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    img = img.transpose((2, 0, 1))[::-1] 
    img = np.ascontiguousarray(img).astype(np.uint8) 
    img = np.expand_dims(img, axis=0)
    
    return img, (h0, w0)

def load_gt_boxes_from_seg(label_path, eval_size=(640, 640)):
    """
    [핵심] Segmentation 폴리곤 라벨을 읽어와서 Box 형태(x1, y1, x2, y2)로 변환합니다.
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) < 5: continue
            
            cls_id = int(data[0])
            # 1. 0~1로 정규화된 폴리곤 좌표를 읽어옵니다.
            coords = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
            
            # 2. 평가 해상도(640x640)에 맞게 스케일을 조정합니다.
            x_coords = coords[:, 0] * eval_size[1]
            y_coords = coords[:, 1] * eval_size[0]
            
            # 3. 폴리곤의 최소/최대 좌표를 추출하여 Bounding Box를 만듭니다.
            x1, y1 = np.min(x_coords), np.min(y_coords)
            x2, y2 = np.max(x_coords), np.max(y_coords)
            
            gt_boxes.append({"class": cls_id, "box": [x1, y1, x2, y2]})
            
    return gt_boxes

def calculate_box_iou(box1, box2):
    """두 개의 Bounding Box 간의 IoU(Intersection over Union)를 계산합니다."""
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0

def evaluate_image(pred_boxes, gt_boxes, iou_thres=0.5):
    TP, FP, FN = 0, 0, len(gt_boxes)
    if pred_boxes is None or len(pred_boxes) == 0:
        return 0, 0, FN
        
    matched_gt_idx = set()
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt_idx: continue
            
            iou = calculate_box_iou(pred_box, gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= iou_thres:
            TP += 1
            FN -= 1
            matched_gt_idx.add(best_gt_idx)
        else:
            FP += 1
            
    return TP, FP, FN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./best.mxq")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--label-dir", type=str, required=True)
    parser.add_argument("--conf-thres", type=float, default=0.35)
    parser.add_argument("--box-iou-thres", type=float, default=0.45)
    args = parser.parse_args()

    # NPU 로드
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = Yolo26SegPostProcessE2E(conf_thres=args.conf_thres)

    total_TP, total_FP, total_FN = 0, 0, 0
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))
    num_images = len(img_paths)

    print(f"Starting Bounding Box evaluation on {num_images} images...")

    for i, img_path in enumerate(img_paths):
        # 1. 전처리 및 Box 정답지 로드
        img_tensor, _ = preprocess_yolo(img_path)
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(args.label_dir, os.path.splitext(img_filename)[0] + ".txt")
        
        gt_boxes = load_gt_boxes_from_seg(label_path, eval_size=(640, 640))
        
        # 2. 모델 추론
        outputs = model.infer([img_tensor])
        
        # 3. 예측 Box 획득 (masks는 사용하지 않음)
        pred_boxes, _ = postprocess(outputs, img_shape=(640, 640))
        
        if pred_boxes is not None and hasattr(pred_boxes, 'cpu'):
            pred_boxes = pred_boxes.cpu().numpy()
            
        # 4. Box IoU 기반 성능 평가
        TP, FP, FN = evaluate_image(pred_boxes, gt_boxes, args.box_iou_thres)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            print(f"Progress: [{i+1}/{num_images}] | TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
        
    model.dispose()

    # 최종 결과 계산
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f"Detection Results (Box IoU >= {args.box_iou_thres}) for {num_images} images:")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
