# [4] 소스 기반 수정
CLASSES = ("car",)

# BGR 형식 (예: 파란색)
DET_PALETTE = [(255, 0, 0)] 

def get_coco_class_num() -> int:
    return len(CLASSES)

def get_coco_label(idx: int) -> str:
    return CLASSES[idx]

def get_coco_det_palette(idx: int) -> tuple:
    return DET_PALETTE[idx]
