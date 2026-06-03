import os
import glob
import argparse
import cv2
import numpy as np
import qbruntime

# 기존에 작성하신 후처리 모듈 임포트
from postprocess import Yolo26SegPostProcessE2E

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    """추론용 이미지 전처리 (inference_mxq.py와 동일하게 유지)"""
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
    
    # HWC to CHW
    img = img.transpose((2, 0, 1))[::-1] 
    # [수정] NPU 모델이 Uint8을 지원하므로 float32 변환 및 /255.0 생략
    img = np.ascontiguousarray(img).astype(np.uint8) 
    img = np.expand_dims(img, axis=0)
    
    return img, (h0, w0)

def load_gt_masks_scaled(label_path, eval_size=(640, 640)):
    """
    [성능 최적화] 원본 크기가 아닌 640x640 스케일에서 정답 마스크를 생성합니다.
    """
    gt_masks = []
    if not os.path.exists(label_path):
        return gt_masks
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split()
            if len(data) < 5: continue
            
            cls_id = int(data[0])
            coords = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
            
            # YOLO 포맷(0~1) 좌표를 직접 640x640 스케일로 변환
            coords[:, 0] *= eval_size[1]
            coords[:, 1] *= eval_size[0]
            pts = coords.astype(np.int32)
            
            mask = np.zeros(eval_size, dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
            gt_masks.append({"class": cls_id, "mask": mask})
            
    return gt_masks

def calculate_mask_iou(mask_pred, mask_gt):
    """두 이진 마스크 간의 IoU 계산 (640x640 해상도에서 수행되어 매우 빠름)"""
    # 픽셀 단위 비트 연산 활용
    intersection = np.count_nonzero(np.logical_and(mask_pred, mask_gt))
    union = np.count_nonzero(np.logical_or(mask_pred, mask_gt))
    return intersection / union if union > 0 else 0.0

def evaluate_image(pred_masks, gt_masks, mask_iou_thres=0.5):
    TP, FP, FN = 0, 0, len(gt_masks)
    if pred_masks is None or len(pred_masks) == 0:
        return 0, 0, FN
        
    matched_gt_idx = set()
    for pred_mask in pred_masks:
        best_iou = 0.0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_masks):
            if i in matched_gt_idx: continue
            
            iou = calculate_mask_iou(pred_mask, gt["mask"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
                
        if best_iou >= mask_iou_thres:
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
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--mask-iou-thres", type=float, default=0.50)
    args = parser.parse_args()

    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    postprocess = Yolo26SegPostProcessE2E(conf_thres=args.conf_thres)

    total_TP, total_FP, total_FN = 0, 0, 0
    img_paths = sorted(glob.glob(os.path.join(args.img_dir, "*.*")))
    num_images = len(img_paths)

    print(f"Starting evaluation on {num_images} images...")

    for i, img_path in enumerate(img_paths):
        # 1. 전처리 및 정답지 로드 (640x640 기준)
        img_tensor, _ = preprocess_yolo(img_path)
        img_filename = os.path.basename(img_path)
        label_path = os.path.join(args.label_dir, os.path.splitext(img_filename)[0] + ".txt")
        gt_masks = load_gt_masks_scaled(label_path, (640, 640))
        
        # 2. NPU 추론 및 후처리
        outputs = model.infer([img_tensor])
        boxes, masks = postprocess(outputs, img_shape=(640, 640)) # masks는 (N, 640, 640)
        
        # 3. 예측 마스크 이진화
        pred_masks_binary = []
        if masks is not None:
            if hasattr(masks, 'cpu'): masks = masks.cpu().numpy()
            for mask in masks:
                pred_masks_binary.append((mask > 0.5).astype(np.uint8))
        
        # 4. 성능 평가
        TP, FP, FN = evaluate_image(pred_masks_binary, gt_masks, args.mask_iou_thres)
        total_TP += TP
        total_FP += FP
        total_FN += FN

        # 진행 상황 출력
        if (i + 1) % 10 == 0 or (i + 1) == num_images:
            print(f"Progress: [{i+1}/{num_images}] | TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
        
    model.dispose()

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print(f"Results for {num_images} images:")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()