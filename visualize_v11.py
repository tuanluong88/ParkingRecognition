import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
import itertools
import os
import inspect
from pathlib import Path
from typing import List
import logging

CLASSES = ('occupied', 'empty')

DET_PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208)]

POSE_PALETTE = [[255, 128, 0], [255, 153, 51], [255, 178, 102],
            [230, 230, 0], [255, 153, 255], [153, 204, 255],
            [255, 102, 255], [255, 51, 255], [102, 178, 255],
            [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102],
            [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
            [255, 255, 255]]

log_level_dict = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
    4: logging.CRITICAL
}

def set_logger(log_level: int):
    logging.basicConfig(level=log_level_dict[log_level], format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def get_coco_class_num() -> int:
    return len(CLASSES)

def get_coco_label(idx: int) -> str:
    type_check(idx, int)
    return CLASSES[idx]

def get_coco_det_palette(idx: int) -> tuple:
    type_check(idx, int)
    return DET_PALETTE[idx]

def get_coco_pose_palette() -> tuple:
    return POSE_PALETTE


def compute_ratio_pad(input_shape, img_shape, ratio_pad = None):
    """ Compute ratio and pad which were used to resize image to input_shape """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(input_shape[0] / img_shape[0], input_shape[1] / img_shape[1])  # gain  = old / new
        pad = (input_shape[1] - int(img_shape[1]*gain+0.5)) / 2, (input_shape[0] - int(img_shape[0]*gain+0.5)) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    return gain, pad
    
    
def clip_coords(boxes, img_shape):
    """ Clip bounding xyxy bounding boxes to image shape (height, width) """
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """ Rescale coords (xyxy) from img1_shape to img0_shape """
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """ Rescale coords (xyxy) from img1_shape to img0_shape """
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def scale_coords_kpts(img1_shape, coords, img0_shape, ratio_pad=None, step=2):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain, pad = compute_ratio_pad(img1_shape, img0_shape, ratio_pad)
    
    coords[:, 0::step] -= pad[0]  # x padding
    coords[:, 1::step] -= pad[1]  # y padding
    coords[:, 0::step] /= gain
    coords[:, 1::step] /= gain
    coords[:, 0::step].clamp_(0, img0_shape[1])
    coords[:, 1::step].clamp_(0, img0_shape[0])
    return coords


def draw_boxes(img, xyxy, desc, cls_color):
    """ plot bounding boxes on image """
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()

    cv2.rectangle(img, (x1,y1), (x2, y2), cls_color, thickness=tl, lineType=cv2.LINE_AA)

    tf = max(tl - 1, 1)  # font thickness
    #cv2.putText(img, desc, (x1, y1 - 2), 0, tl / 2, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def draw_landmarks(img, landmarks):
    """ mark landmarks on face """
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)
    
    return img

def draw_masks(img, masks, colors, alpha = 0.5):
    gain, pad = compute_ratio_pad(masks.shape[1:], img.shape)
    pad = (int(pad[0]), int(pad[1]))
    top, left = pad[1], pad[0]
    bottom, right = masks.shape[1] - pad[1], masks.shape[2] - pad[0]

    # revert mask resolution to original image resolution
    masks = masks[:, top:bottom, left:right]
    masks = F.interpolate(masks.unsqueeze(0), scale_factor=1/gain, mode='bilinear', align_corners=False).squeeze(0)
    masks = masks.gt_(0.5)

    colors = np.array(colors, dtype=np.float32) / 255
    colors = colors[:, None, None]  # shape(n,1,1,3)
    masks = np.expand_dims(masks, 3)  # shape(n,h,w,1)
    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

    img = img.astype(np.float32) / 255
    img = img * inv_alph_masks[-1] + mcs
    img = (img * 255).astype(np.uint8)
    
    return img

def draw_kpts(im, kpts, steps, palette, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array(palette)

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)
    return im

def type_check(obj, class_input: List):
    error_msg = "class_input must be a class or list of class, but got "

    if isinstance(class_input, list):
        for class_ in class_input:
            if not inspect.isclass(class_):
                raise TypeError(f"class_input must be a class or list of class, but got {type(class_)} in the list.")

        if not any([isinstance(obj, class_) for class_ in class_input]):
            class_list = set([t.__name__ for t in class_input])
            raise TypeError(f"Expected one of {class_list} input, but got: {type(obj)}.")
    else:
        if not inspect.isclass(class_input):
            raise TypeError(f"class_input must be a class or list of class, but got {type(class_)}.")
        
        if not isinstance(obj, class_input):
            raise TypeError(f"Got unexpected type: {type(obj)}.")

class BaseVisualizer(ABC):
    def __init__(self, dataset: str='coco'):
        self.dataset = dataset
        self.writer = None
        
        if self.dataset == "imagenet":
            self.get_label = self.get_imagenet_label
        elif self.dataset == "coco":
            self.get_label = self.get_coco_label
            self.get_color = self.get_coco_det_palette
            self.get_pose_palette = self.get_coco_pose_palette
        elif self.dataset == 'pascal_voc':
            self.get_label = self.get_pascal_voc_label
            self.get_color = self.get_pascal_voc_palette
        elif self.dataset == 'bird500':
            self.get_label = get_bird500_label
        else:
            raise NotImplementedError(f"Got unsupported dataset: ", self.dataset)
    
    @abstractmethod
    def save(self, out_post_processed, **kwargs):
        ...
    
    def get_imagenet_label(self, idx: int):
        return get_imagenet_label(idx)

    def get_pascal_voc_label(self, idx: int):
        return get_pascal_voc_label(idx)

    def get_pascal_voc_palette(self, idx: int):
        return get_pascal_voc_palette(idx)

    def get_coco_label(self, idx: int):
        return get_coco_label(idx)
    
    def get_coco_det_palette(self, idx: int):
        return get_coco_det_palette(idx)
    
    def get_coco_pose_palette(self):
        return get_coco_pose_palette()
    
    def set_video_writer(self, output_path: str, fps: float, video_size: Tuple[int, int]) -> None:
        self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_size)
        self.fps = fps
    
    def release(self):
        if self.writer is None:
            print("Video writer is not set yet.")
        else:
            self.writer.release()
            
class YoloVisualizer(BaseVisualizer):
    def __init__(self) -> None:
        super().__init__("coco")
        self.model_input_size = [640, 640]
    
    def save(self, out_post_processed: List[torch.Tensor], input_path: str, output_path: str=None):
        if isinstance(input_path, str):
            img = cv2.imread(input_path)
        elif isinstance(input_path, np.ndarray):
            img = input_path
        else:
            raise TypeError(f"Got unsupported input type: {type(input_path)}")
        
        # --- [수정된 부분] 탐지된 객체가 없는 경우 처리 ---
        if out_post_processed is None or len(out_post_processed) == 0:
            print(f"No objects detected in {input_path}")
            if output_path is not None:
                cv2.imwrite(output_path, img) # 원본 이미지 그대로 저장
            return
        # -------------------------------------------------

        det = out_post_processed[0]
        num_det = det.shape[0]
        
        det[:, :4] = scale_coords(self.model_input_size, det[:, :4], img.shape).round()
        for j in range(num_det):
            xyxy = det[j, :4].view(-1).tolist()
            conf = det[j, 4].cpu().numpy()
            cls = det[j, 5].cpu().numpy().item()
            cls_name = self.get_label(int(cls))
            cls_color = self.get_color(int(cls))
            desc = f'{cls_name}: {round(conf.item(),3)}'
            img = draw_boxes(img, xyxy, desc, cls_color)
            logging.info(f"Predicted object: {desc}")
        
        delay = 0
        if output_path is not None: # image demo
            cv2.imwrite(output_path, img)