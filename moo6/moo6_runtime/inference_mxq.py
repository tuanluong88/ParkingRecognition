import os
import argparse
import cv2
import numpy as np
import qbruntime
from postprocess import Yolo26SegPostProcessE2E
from visualize import YoloVisualizer

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # 1. 이미지 읽기 및 색상 변환
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. 리사이즈 (Letterbox 비율 유지)
    h0, w0 = img.shape[:2]
    r = min(img_size[0] / h0, img_size[1] / w0)
    new_unpad = int(round(w0 * r)), int(round(h0 * r))
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 3. 패딩 (중앙 정렬)
    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]
    top, bottom = int(round(dh // 2)), int(round(dh - dh // 2))
    left, right = int(round(dw // 2)), int(round(dw - dw // 2))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # --- 수정된 핵심 부분 ---
    # 4. 정규화(/255.0)를 생략하고 uint8 상태를 유지합니다.
    # 5. [H, W, C] -> [1, C, H, W] 순서로 변경
    img = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
    
    # 6. 데이터 타입을 uint8로 확실히 지정하여 반환합니다.
    return img.astype(np.uint8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./best.mxq", help="Path to compiled MXQ model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-path", type=str, default="./output_result.jpg", help="Path to output image")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    # NPU 초기화 및 모델 로드
    acc = qbruntime.Accelerator()
    mc = qbruntime.ModelConfig()
    mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
    model = qbruntime.Model(args.model_path, mc)
    model.launch(acc)

    # YOLOv26 E2E 전용 후처리기 및 시각화 객체 생성
    postprocess = Yolo26SegPostProcessE2E(conf_thres=args.conf_thres)
    visualizer = YoloVisualizer(model_input_size=(640, 640))

    img_tensor = preprocess_yolo(args.image_path)
    
    # NPU 추론 실행
    outputs = model.infer([img_tensor])

    # 후처리 연산 (디코딩 및 NMS)
    boxes, masks = postprocess(outputs, img_shape=(640, 640))

    if boxes is not None and len(boxes) > 0:
        # 시각화 및 저장
        visualizer.draw(args.image_path, [boxes], masks=[masks], output_path=args.output_path)
        print(f"================================")
        print(f"추론 완료! {len(boxes)}개의 객체가 탐지되었습니다.")
        print(f"결과가 {args.output_path} 에 저장되었습니다.")
    else:
        print("탐지된 객체가 없거나 Confidence Threshold에 미치지 못했습니다.")