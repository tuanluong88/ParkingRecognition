import os
import glob
from argparse import ArgumentParser

import cv2
import numpy as np
import qbruntime
from postprocess import YoloPostProcessAnchorless

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

def load_yolo_seg_labels(label_path, img_w, img_h):
    """
    Loads YOLO segmentation format labels (class_id x1 y1 x2 y2 ...) 
    and converts them to absolute bounding box coordinates.
    """
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # A valid segmentation label should have at least class_id + 6 points (a triangle)
            if len(parts) >= 7:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                
                # Extract x and y coordinates (alternating)
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                
                # Find the min and max values to create the bounding box
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # Convert normalized coordinates (0.0 to 1.0) to absolute pixel values
                x1 = x_min * img_w
                y1 = y_min * img_h
                x2 = x_max * img_w
                y2 = y_max * img_h
                
                gt_boxes.append({'class_id': class_id, 'box': [x1, y1, x2, y2], 'matched': False})
    return gt_boxes

def evaluate_predictions(predictions, gt_boxes, iou_threshold=0.5):
    """Calculate True Positives, False Positives, and False Negatives for a single image."""
    TP, FP = 0, 0
    
    # Sort predictions by confidence in descending order
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
            # We found a match
            gt_boxes[best_gt_idx]['matched'] = True
            TP += 1
        else:
            # No matching ground truth found (False Positive)
            FP += 1
            
    # Any ground truth that wasn't matched is a False Negative
    FN = sum(1 for gt in gt_boxes if not gt['matched'])
    
    return TP, FP, FN

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate MXQ model performance on a dataset with segmentation labels")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the compiled MXQ model")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Path to the dataset directory (must contain 'images' and 'labels' folders)")
    parser.add_argument("--conf-thres", type=float, default=0.35, help="Confidence threshold for predictions")
    parser.add_argument("--nms-iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--eval-iou-thres", type=float, default=0.45, help="IoU threshold for evaluating True Positives")
    
    args = parser.parse_args()

    # Setup qbruntime
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = YoloPostProcessAnchorless(args.conf_thres, args.nms_iou_thres)

    image_paths = glob.glob(os.path.join(args.dataset_dir, "images", "*.*"))
    label_dir = os.path.join(args.dataset_dir, "labels")

    total_TP = 0
    total_FP = 0
    total_FN = 0

    print(f"Starting evaluation on {len(image_paths)} images...")

    for img_path in image_paths:
        img, (h0, w0) = preprocess_yolo(img_path)
        outputs = model.infer([img])
        
        result = postprocess(outputs) 
        
        # --- 새롭게 추가/수정된 코드 ---
        # 1. 배치(batch)에서 첫 번째 이미지의 예측 결과만 가져옵니다.
        image_preds = result[0] if len(result) > 0 else []
        
        # 2. 출력값이 PyTorch 텐서 형태일 경우, 스칼라 변환 에러를 막기 위해 NumPy 배열로 변환합니다.
        if hasattr(image_preds, 'cpu'):
            image_preds = image_preds.cpu().numpy()
        # -------------------------------
        
        img_filename = os.path.basename(img_path)
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        
        gt_boxes = load_yolo_seg_labels(label_path, w0, h0)
        
        # 3. 'result' 대신 처리된 'image_preds'를 전달합니다.
        TP, FP, FN = evaluate_predictions(image_preds, gt_boxes, args.eval_iou_thres)
        
        total_TP += TP
        total_FP += FP
        total_FN += FN

    model.dispose()

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Performance Results ---")
    print(f"Total True Positives (TP) : {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print("---------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1_score:.4f}")
