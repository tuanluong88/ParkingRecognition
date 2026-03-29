import os
import json
import cv2
import torch
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 이전에 수정한 utils_v11 파일에서 함수 임포트
from utils_v11 import preprocess_yolo, YoloPostProcess

# ----------------- 설정 부분 -----------------
# 파일 경로를 본인 환경에 맞게 수정해주세요.
YOLO_ONNX_PATH = '/home/rise/Sim/yolov11/modified_best_yolo200.onnx'
TEST_IMG_DIR = '/home/rise/Sim/pyCode/ParkingLot/test/' # 테스트 이미지가 있는 폴더
ANNOTATION_FILE = '/home/rise/Sim/pyCode/ParkingLot/test/_annotations.coco.json' # 테스트 정답 json
# ---------------------------------------------

def scale_coords(img1_shape, coords, img0_shape):
    """ 추론된 좌표(640x640 기준)를 원본 이미지 크기에 맞게 복원하는 함수 """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = (img1_shape[1] - img0_shape[1] * gain) / 2
    pad_y = (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain

    # 원본 이미지 바깥으로 나가는 좌표 잘라내기
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords

def main():
    print("Loading Ground Truth JSON...")
    cocoGt = COCO(ANNOTATION_FILE)
    image_ids = cocoGt.getImgIds()
    
    # 모델 및 후처리 객체 로드
    model = ort.InferenceSession(YOLO_ONNX_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    postprocess = YoloPostProcess()
    postprocess.nc = 2  # 클래스 2개 강제 할당 (이전 수정사항 반영)
    
    results = []
    print(f"Start evaluating {len(image_ids)} images...")

    for count, img_id in enumerate(image_ids):
        # 1. 이미지 정보 불러오기
        img_info = cocoGt.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_path = os.path.join(TEST_IMG_DIR, file_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found.")
            continue

        # 2. 이미지 전처리 및 추론
        img = preprocess_yolo(img_path)
        img_tensor = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
        outputs = model.run(None, {'images': img_tensor})
        out_post_processed = postprocess.run(outputs)

        # 탐지된 객체가 없으면 스킵
        if out_post_processed is None or len(out_post_processed) == 0:
            continue
            
        det = out_post_processed[0].clone()
        
        # 3. 좌표 스케일 원복
        orig_img = cv2.imread(img_path)
        orig_shape = orig_img.shape
        det[:, :4] = scale_coords([640, 640], det[:, :4], orig_shape).round()
        
        # 4. COCO Results 포맷으로 변환
        for j in range(det.shape[0]):
            x1, y1, x2, y2 = det[j, :4].tolist()
            conf = float(det[j, 4].cpu().numpy())
            cls_idx = int(det[j, 5].cpu().numpy())
            
            # width, height 계산
            w = x2 - x1
            h = y2 - y1
            
            # [중요] YOLO의 class index(0, 1)를 COCO의 category_id(1, 2)에 맵핑
            # _annotations.coco.json 기준: 1=space-empty, 2=space-occupied
            category_id = cls_idx + 1 
            
            results.append({
                "image_id": img_id,
                "category_id": category_id,
                "bbox": [x1, y1, w, h],
                "score": conf
            })
            
        if (count + 1) % 50 == 0:
            print(f"Processed {count + 1}/{len(image_ids)} images...")

    # 5. 결과를 임시 JSON으로 저장 후 평가 진행
    if not results:
        print("No objects detected in the entire dataset!")
        return
        
    print("Writing temporary result file...")
    temp_res_file = 'temp_results.json'
    with open(temp_res_file, 'w') as f:
        json.dump(results, f)
        
    print("Running COCO Evaluation...")
    cocoDt = cocoGt.loadRes(temp_res_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    # 평가 실행 및 요약 출력
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # 임시 파일 삭제
    os.remove(temp_res_file)

if __name__ == '__main__':
    main()