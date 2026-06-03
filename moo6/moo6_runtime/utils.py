import torch
import torch.nn.functional as F

def process_mask_upsample(protos, masks_in, bboxes, shape):
    """
    NPU 출력 Proto와 계수를 곱해 마스크 생성
    """
    c, mh, mw = protos.shape 
    # Matrix Multiplication (N, 32) @ (32, 160*160) -> (N, 160, 160)
    masks = (masks_in @ protos.view(c, -1)).view(-1, mh, mw)
    
    # 640x640 이미지 사이즈로 확대
    masks = F.interpolate(masks[None], shape, mode="bilinear", align_corners=False)[0]
    
    # Bbox 영역으로 마스크 크롭
    masks = crop_mask(masks, bboxes)
    return masks.gt_(0.0).float()

def crop_mask(masks, boxes):
    """마스크를 박스 영역으로 제한"""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1) # [n, 1, 1]
    r = torch.arange(w, device=masks.device)[None, None, :] # [1, 1, w]
    c = torch.arange(h, device=masks.device)[None, :, None] # [1, h, 1]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))