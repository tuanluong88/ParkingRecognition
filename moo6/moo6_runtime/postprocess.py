import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from utils import process_mask_upsample

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """피처맵 규격에 맞춰 앵커 포인트와 스트라이드를 생성하는 함수"""
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class Yolo26SegPostProcessE2E:
    def __init__(self, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def __call__(self, outputs, img_shape=(640, 640)):
        # 1. 스케일별 출력 텐서 그룹화 (20x20, 40x40, 80x80)
        conf_20, box_20, mc_20 = outputs[0], outputs[1], outputs[2]
        conf_40, box_40, mc_40 = outputs[3], outputs[4], outputs[5]
        conf_80, box_80, mc_80 = outputs[6], outputs[7], outputs[8]
        proto = torch.from_numpy(outputs[9])[0] # (32, 160, 160)

        # 2. Flatten 및 Concat 
        def flatten_and_cat(tensors):
            res = []
            for t in tensors:
                t = torch.from_numpy(t)
                res.append(t.view(1, t.shape[1], -1))
            return torch.cat(res, dim=2).transpose(1, 2)

        all_conf = flatten_and_cat([conf_20, conf_40, conf_80]) # (1, 8400, 1)
        all_box = flatten_and_cat([box_20, box_40, box_80])     # (1, 8400, 4)
        all_mc = flatten_and_cat([mc_20, mc_40, mc_80])         # (1, 8400, 32)

        # Confidence에 Sigmoid 적용하여 0~1 확률값으로 변환
        all_conf = all_conf.sigmoid()

        # 3. 바운딩 박스 디코딩 (Distance -> Absolute Pixel Coordinates)
        feats = [torch.from_numpy(box_20), torch.from_numpy(box_40), torch.from_numpy(box_80)]
        anchors, strides = make_anchors(feats, [32, 16, 8])
        
        # 좌상단(x1, y1) 우하단(x2, y2) 좌표 복원
        x1y1 = anchors - all_box[0, :, :2]
        x2y2 = anchors + all_box[0, :, 2:]
        bboxes = torch.cat((x1y1, x2y2), -1) * strides # (8400, 4)

        scores = all_conf[0, :, 0]
        mc = all_mc[0]

        # 4. Confidence 필터링
        valid_idx = scores > self.conf_thres
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        mc = mc[valid_idx]

        if len(bboxes) == 0:
            return None, None

        # 5. NMS (Non-Maximum Suppression) 적용 (중복 박스 제거)
        keep = torchvision.ops.nms(bboxes, scores, self.iou_thres)
        bboxes = bboxes[keep]
        scores = scores[keep]
        mc = mc[keep]

        # Class ID 생성 (단일 클래스인 경우 0)
        cls_ids = torch.zeros_like(scores)

        # Visualizer 규격에 맞게 병합: [x1, y1, x2, y2, conf, cls]
        det_boxes = torch.cat([bboxes, scores.unsqueeze(1), cls_ids.unsqueeze(1)], dim=1)

        # 6. 마스크 생성 연산
        masks = process_mask_upsample(proto, mc, bboxes, img_shape)

        return det_boxes.numpy(), masks.numpy()