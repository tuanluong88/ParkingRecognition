CLASSES = (
    "car",
)

DET_PALETTE = [
    (0, 142, 0),  # 기존 'car' 클래스의 색상입니다. 원하시는 RGB 값으로 변경하셔도 됩니다.
]


def get_coco_class_num() -> int:
    return len(CLASSES)


def get_coco_label(idx: int) -> str:
    return CLASSES[idx]


def get_coco_det_palette(idx: int) -> tuple:
    return DET_PALETTE[idx]