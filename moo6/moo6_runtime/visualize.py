import cv2
import numpy as np
import torch

def scale_boxes(img1_shape, coords, img0_shape):
    """입력 이미지(640x640) 비율에 맞추어 모델 좌표를 원본 이미지 좌표계로 스케일링 (Numpy 버전)"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad[0]  # x padding 제거
    coords[:, [1, 3]] -= pad[1]  # y padding 제거
    coords[:, :4] /= gain        # 원본 스케일로 복구
    
    # .clamp() 대신 Numpy의 .clip() 사용
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, img0_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, img0_shape[0])
    return coords

class YoloVisualizer:
    def __init__(self, model_input_size=(640, 640)):
        self.model_input_size = model_input_size
        self.palette = np.random.randint(0, 255, (80, 3), dtype=np.uint8) # 랜덤 컬러 팔레트 (COCO 80)

    def draw(self, input_path, out_post_processed, masks=None, output_path=None):
        img = cv2.imread(input_path)
        orig_shape = img.shape[:2]
        
        # out_post_processed는 리스트 형태 [batch_result]
        det = out_post_processed[0]
        if det is not None and len(det) > 0:
            # 1. 박스 스케일링 (이미 Numpy 배열임)
            det[:, :4] = scale_boxes(self.model_input_size, det[:, :4], orig_shape)
            
            col_list = []
            for j in range(len(det)):
                # .cpu().numpy() 제거
                xyxy = det[j, :4].astype(int)
                conf = det[j, 4]
                cls = int(det[j, 5])

                color = [int(c) for c in self.palette[cls]]
                col_list.append(color)

                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(img, f"cls:{cls} {conf:.2f}", (xyxy[0], xyxy[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if masks is not None and masks[0] is not None:
                img = self.add_mask(img, masks[0], col_list, orig_shape)

        if output_path is not None:
            cv2.imwrite(output_path, img)

        return img

    def add_mask(self, img, masks, colors, orig_shape):
        """마스크를 원본 이미지 스케일에 맞추고 색상을 입히는 함수 (Numpy/OpenCV 버전)"""
        # masks shape: (N, 640, 640) - 후처리에서 이미 640으로 upsample됨
        # 하지만 원본 이미지(orig_shape)가 640이 아닐 경우를 대비해 최종 리사이즈 수행
        
        full_mask = np.zeros(img.shape, dtype=np.uint8)
        
        for i, mask in enumerate(masks):
            # 1. 마스크를 원본 이미지 크기로 리사이즈
            # (640, 640) -> (W, H) of original image
            mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # 2. 이진화 (0.5 기준)
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # 3. 색상 입히기
            color = colors[i]
            for c in range(3):
                full_mask[:, :, c] += (mask_binary * color[c]).astype(np.uint8)

        # 4. 원본 이미지와 마스크 합성 (Alpha blending)
        alpha = 0.5
        img = cv2.addWeighted(img, 1.0, full_mask, alpha, 0)
        
        return img