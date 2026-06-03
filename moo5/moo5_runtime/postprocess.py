import numpy as np
import torch
import torch.nn.functional as F
from utils import process_mask_upsample

class Yolo26SegPostProcessE2E:
    def __init__(self, conf_thres=0.25):
        self.conf_thres = conf_thres

    def __call__(self, outputs, img_shape=(640, 640)):
        # 1. NPU 출력 텐서들을 용도별로 그룹화
        # 20x20 스케일
        conf_20, box_20, mc_20 = outputs[0], outputs[1], outputs[2]
        # 40x40 스케일
        conf_40, box_40, mc_40 = outputs[3], outputs[4], outputs[5]
        # 80x80 스케일
        conf_80, box_80, mc_80 = outputs[6], outputs[7], outputs[8]
        # Proto Mask
        proto = torch.from_numpy(outputs[9])[0] # (32, 160, 160)

        # 2. 각 피처맵을 Flatten 하여 하나로 합침 (CPU 연산)
        def flatten_and_cat(tensors):
            res = []
            for t in tensors:
                t = torch.from_numpy(t)
                res.append(t.view(1, t.shape[1], -1)) # (1, C, H*W)
            return torch.cat(res, dim=2).transpose(1, 2) # (1, 8400, C)

        all_conf = flatten_and_cat([conf_20, conf_40, conf_80]) # (1, 8400, 1)
        all_box = flatten_and_cat([box_20, box_40, box_80])   # (1, 8400, 4)
        all_mc = flatten_and_cat([mc_20, mc_40, mc_80])      # (1, 8400, 32)

        # 3. 전체 예측값 구성 (Box + Conf + Class(0으로 가정) + Mask Coef)
        # YOLOv26 E2E 모델은 보통 상위 300개를 추출하지만, 여기서는 모든 후보군에 대해 Conf 필터링 진행
        preds = torch.cat([all_box, all_conf, torch.zeros_like(all_conf), all_mc], dim=2)[0] # (8400, 38)

        # 4. Confidence Threshold 적용
        conf_scores = preds[:, 4]
        valid_idx = conf_scores > self.conf_thres
        pred_valid = preds[valid_idx]

        if len(pred_valid) == 0:
            return [(None, None)]

        # 5. 분리 및 후처리
        boxes = pred_valid[:, :4]
        scores = pred_valid[:, 4]
        classes = pred_valid[:, 5]
        mask_coefs = pred_valid[:, 6:] # (N, 32)

        # 마스크 생성 유틸리티 호출
        masks = process_mask_upsample(proto, mask_coefs, boxes, img_shape)
        
        det = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
        return [(det, masks)]