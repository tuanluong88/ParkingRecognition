import os
import json
import cv2
import yaml
import torch
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 이전 수정사항 반영된 utils_v11 파일 임포트
from utils_v11 import preprocess_yolo, YoloPostProcess

# ================= 1. 경로 설정 부분 (직접 수정) =================
YOLO_ONNX_PATH = '/home/rise/Sim/yolov11/modified_best_yolo1000.onnx'
DATA_YAML_PATH = '/home/rise/Sim/pyCode/SKKUdata/data/data.yaml'

# 실제 데이터가 있는 폴더 경로를 정확히 입력하세요.
TEST_IMG_DIR = '/home/rise/Sim/pyCode/SKKUdata/data/test/images'
TEST_LABEL_DIR = '/home/rise/Sim/pyCode/SKKUdata/data/test/labels'
# =============================================================

def scale_coords(img1_shape, coords, img0_shape):
    """추론된 좌표(640x640)를 원본 이미지 크기에 맞게 복원"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain

    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords

def load_yolo_labels_to_coco(test_img_dir, test_label_dir, class_names):
    """YOLO txt 라벨들을 COCO 객체 형식으로 변환"""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(class_names)]
    }
    
    img_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    ann_id = 0
    
    for i, img_file in enumerate(img_files):
        img_path = os.path.join(test_img_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, _ = img.shape
        
        coco_data["images"].append({
            "id": i,
            "file_name": img_file,
            "width": w,
            "height": h
        })
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(test_label_dir, label_file)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.split()))
                    cls = int(parts[0])
                    coords = parts[1:]
                    
                    # 폴리곤(점 여러 개)인 경우 최소/최대 좌표를 찾아 Bbox 생성
                    xs = coords[0::2] # 홀수 번째 (x좌표들)
                    ys = coords[1::2] # 짝수 번째 (y좌표들)
                    
                    x1, y1 = min(xs), min(ys)
                    x2, y2 = max(xs), max(ys)
                    
                    abs_w = (x2 - x1) * w
                    abs_h = (y2 - y1) * h
                    abs_x1 = x1 * w
                    abs_y1 = y1 * h
                    
                    coco_data["annotations"].append({
                        "id": ann_id,
                        "image_id": i,
                        "category_id": cls,
                        "bbox": [abs_x1, abs_y1, abs_w, abs_h],
                        "area": abs_w * abs_h,
                        "iscrowd": 0
                    })
                    ann_id += 1
                    
    return coco_data

def main():
    # 1. 클래스 정보 로드
    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: data.yaml not found at {DATA_YAML_PATH}")
        return
        
    with open(DATA_YAML_PATH, 'r') as f:
        data_info = yaml.safe_load(f)
    class_names = data_info['names']

    # 2. 경로 존재 확인
    if not os.path.exists(TEST_IMG_DIR):
        print(f"Error: Image directory not found: {TEST_IMG_DIR}")
        return
    if not os.path.exists(TEST_LABEL_DIR):
        print(f"Error: Label directory not found: {TEST_LABEL_DIR}")
        return

    # 3. 정답 데이터 로드 및 COCO 객체 생성
    print("Converting YOLO labels to COCO format...")
    gt_dict = load_yolo_labels_to_coco(TEST_IMG_DIR, TEST_LABEL_DIR, class_names)
    
    with open('tmp_gt.json', 'w') as f:
        json.dump(gt_dict, f)
    cocoGt = COCO('tmp_gt.json')
    
    # 4. 모델 로드 (onnxruntime)
    print(f"Loading Model: {YOLO_ONNX_PATH}")
    model = ort.InferenceSession(YOLO_ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    postprocess = YoloPostProcess()
    postprocess.nc = len(class_names)
    
    results = []
    image_ids = cocoGt.getImgIds()
    print(f"Start evaluating {len(image_ids)} images...")

    # 5. 추론 및 결과 수집
    for count, img_id in enumerate(image_ids):
        img_info = cocoGt.loadImgs(img_id)[0]
        img_path = os.path.join(TEST_IMG_DIR, img_info['file_name'])
        
        img = preprocess_yolo(img_path)
        img_tensor = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
        outputs = model.run(None, {'images': img_tensor})
        out_post_processed = postprocess.run(outputs)

        if out_post_processed is None or len(out_post_processed) == 0:
            continue
            
        det = out_post_processed[0].clone()
        orig_img = cv2.imread(img_path)
        det[:, :4] = scale_coords([640, 640], det[:, :4], orig_img.shape).round()
        
        for j in range(det.shape[0]):
            x1, y1, x2, y2 = det[j, :4].tolist()
            conf = float(det[j, 4].cpu().numpy())
            cls_idx = int(det[j, 5].cpu().numpy())
            
            results.append({
                "image_id": img_id,
                "category_id": cls_idx,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf
            })
        
        if (count + 1) % 50 == 0:
            print(f"Progress: {count + 1}/{len(image_ids)}")

    # 6. 성능 지표(mAP) 계산
    if not results:
        print("No objects detected!")
    else:
        with open('tmp_dt.json', 'w') as f:
            json.dump(results, f)
        
        cocoDt = cocoGt.loadRes('tmp_dt.json')
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        if os.path.exists('tmp_dt.json'):
            os.remove('tmp_dt.json')
    
    if os.path.exists('tmp_gt.json'):
        os.remove('tmp_gt.json')

if __name__ == '__main__':
    main()