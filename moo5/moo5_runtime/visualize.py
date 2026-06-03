import cv2
import numpy as np
import torch

def scale_boxes(img1_shape, coords, img0_shape):
    """입력 이미지 비율에 맞추어 모델 좌표를 원본 이미지 좌표계로 스케일링"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])
    return coords

class YoloVisualizer:
    def __init__(self, model_input_size=(640, 640)):
        self.model_input_size = model_input_size
        self.palette = np.random.randint(0, 255, (80, 3), dtype=np.uint8) # 랜덤 컬러 팔레트 (COCO 80)

    def draw(self, input_path, out_post_processed, masks=None, output_path=None):
        img = cv2.imread(input_path)
        orig_shape = img.shape[:2]

        det = out_post_processed[0]
        if det is None:
            if output_path: cv2.imwrite(output_path, img)
            return img

        # 박스 좌표를 원본 이미지 스케일로 조정
        det[:, :4] = scale_boxes(self.model_input_size, det[:, :4], orig_shape)

        num_det = det.shape[0]
        col_list = []

        for j in range(num_det):
            xyxy = det[j, :4].cpu().numpy().astype(int)
            conf = det[j, 4].cpu().numpy()
            cls = int(det[j, 5].cpu().item())
            color = self.palette[cls].tolist()
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
        """마스크를 원본 이미지 스케일에 맞추고 색상을 입히는 함수"""
        # 마스크를 원본 이미지 사이즈로 resize
        gain = min(self.model_input_size[0] / orig_shape[0], self.model_input_size[1] / orig_shape[1])
        pad = (self.model_input_size[1] - orig_shape[1] * gain) / 2, (self.model_input_size[0] - orig_shape[0] * gain) / 2
        
        top, bottom = int(pad[1]), int(self.model_input_size[0] - pad[1])
        left, right = int(pad[0]), int(self.model_input_size[1] - pad[0])
        
        masks = masks[:, top:bottom, left:right]
        masks = torch.nn.functional.interpolate(masks[None], size=orig_shape, mode="bilinear", align_corners=False)[0]
        masks = masks.cpu().numpy() > 0.5
        
        mask_overlay = np.zeros_like(img)
        for i, mask in enumerate(masks):
            color = np.array(colors[i], dtype=np.uint8)
            mask_overlay[mask] = color

        img = cv2.addWeighted(img, 1.0, mask_overlay, 0.5, 0)
        return img