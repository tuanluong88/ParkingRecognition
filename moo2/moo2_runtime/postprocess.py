from multiprocessing.pool import ThreadPool

import numpy as np
import torch
from utils import (
    DEVICE,
    DFL,
    NUM_THREADS,
    dist2bbox,
    invsigmoid,
    make_anchors,
    non_max_suppression,
)
# Modified
from coco import get_coco_class_num


class YoloPostProcessAnchorless:
    """YOLO object detection postprocess for anchorless models."""

    def __init__(self, conf_thres=0.5, iou_thres=0.5):
        self.imh = self.imw = 640
        # Modified
        # self.nc = 80
        self.nc = get_coco_class_num()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.dfl = DFL(self.reg_max)

        self.nl = 3
        self.stride = [2 ** (3 + i) for i in range(self.nl)]
        self.anchors, self.strides = (xx.transpose(0, 1) for xx in make_anchors(self.imh, self.imw, self.stride, 0.5))
        self.n_extra = 0
        self.device = DEVICE
        self.invconf_thres = invsigmoid(self.conf_thres)

    def check_input(
        self,
        x,
    ) -> list[torch.Tensor]:
        """Check input type and convert to ListTensor

        Args:
            x (TensorLike): Input

        Raises:
            NotImplementedError: Input type not supported.

        Returns:
            List[torch.Tensor]
        """
        if isinstance(x, np.ndarray):
            return [torch.from_numpy(x).to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, np.ndarray) for xi in x):
            return [torch.from_numpy(xi).to(self.device) for xi in x]
        if isinstance(x, torch.Tensor):
            return [x.to(self.device)]
        if isinstance(x, list) and all(isinstance(xi, torch.Tensor) for xi in x):
            return [xi.to(self.device) for xi in x]

        raise NotImplementedError(f"Input type {type(x)} not supported.")

    def __call__(self, x):
        x = self.check_input(x)
        x = self.rearrange_npu_out(x)
        x = self.decode(x)
        x = self.nms(x)
        if x[0].nelement() == 0:
            return None
        return x

    def rearrange_npu_out(self, x):
        y_det = []
        y_cls = []
        for xi in x:  # list of bchw outputs
            if xi.ndim == 3:
                xi = xi[None]
            elif xi.ndim == 4:
                pass
            else:
                raise NotImplementedError(f"Got unsupported ndim for input: {xi.ndim}.")

            if xi.shape[1] == self.reg_max * 4:
                y_det.append(xi)  # (b, 64, 80, 80), (b, 64 ,40, 40), ...
            elif xi.shape[1] == self.nc:
                y_cls.append(xi)  # (b, 80, 80, 80), (b, 80, 40, 40), ...
            else:
                raise ValueError(f"Wrong shape of input: {xi.shape}")

        # sort as box, scores
        y_det = sorted(y_det, key=lambda x: x.numel(), reverse=True)
        y_cls = sorted(y_cls, key=lambda x: x.numel(), reverse=True)

        y = [torch.cat((yi_det, yi_cls), dim=1).flatten(2) for (yi_det, yi_cls) in zip(y_det, y_cls)]

        return y

    def decode(self, x):
        batch_box_cls = torch.cat(x, dim=-1)  # (b, 144, 8400)

        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(self.process_box_cls, batch_box_cls)  # [(*, 84), (*, 84), (*, 84), ...]

        return [self.process_box_cls(box_cls) for box_cls in batch_box_cls]

    def process_box_cls(self, box_cls):
        if self.n_extra == 0:
            ic = torch.amax(box_cls[-self.nc :, :], dim=0) > self.invconf_thres
        else:
            ic = torch.amax(box_cls[-self.nc - self.n_extra : -self.n_extra, :], dim=0) > self.invconf_thres
        box_cls = box_cls[:, ic]  # (144, *)
        if box_cls.numel() == 0:
            return torch.zeros((0, 4 + self.nc + self.n_extra), dtype=torch.float32)  # (0, 84)

        box, scores, extra = torch.split(
            box_cls[None], [self.reg_max * 4, self.nc, self.n_extra], dim=1
        )  # (1, 64, *), (1, 80, *), (1, 32, *)
        dbox = (
            dist2bbox(
                self.dfl(box),
                self.anchors[:, ic],
                xywh=False,
                dim=1,
            )
            * self.strides[:, ic]
        )

        return torch.cat([dbox, scores.sigmoid(), extra], dim=1).squeeze(0).transpose(0, 1)

    def nms(
        self,
        prediction,
        agnostic=False,
        max_det=300,
        max_nms=30000,
        max_wh=7680,
    ):
        """
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L162
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (np.array): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            max_det (int): The maximum number of boxes to keep after NMS.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into NMS.
            max_wh (int): The maximum box width and height in pixels

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= self.conf_thres <= 1, (
            f"Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0"
        )
        assert 0 <= self.iou_thres <= 1, f"Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0"

        def nms_single(x):
            box, conf, mask = torch.split(x, [4, self.nc, self.n_extra], dim=1)

            i, j = torch.nonzero(conf > self.conf_thres).T  # use multi-label as default
            x = torch.cat(
                (box[i], x[i, 4 + j, None], j[:, None].to(torch.float32), mask[i]),
                dim=1,
            )

            if x.numel() == 0:  # no boxes
                return torch.zeros((0, 6 + self.n_extra), dtype=torch.float32)

            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence with descending order and remove excess boxes

            c = x[:, 5:6] * (0.0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = non_max_suppression(boxes, scores, self.iou_thres, max_det)  # NMS

            return x[i, :]

        if self.device.type == "cpu":
            with ThreadPool(NUM_THREADS) as pool:
                return pool.map(nms_single, prediction)

        return [nms_single(p) for p in prediction]

# Modified (line 193-253)
class RTDETRPostProcess:
    def __init__(self, conf_thres=0.25):
        self.conf_thres = conf_thres
        self.nc = get_coco_class_num()

    def __call__(self, x):
        # 1. 300개의 쿼리가 담긴 RT-DETR 최종 출력 텐서 찾기
        pred_boxes = None
        pred_scores = None
        pred_concat = None

        for xi in x:
            if not isinstance(xi, torch.Tensor):
                xi = torch.from_numpy(xi)
            
            # 객체 쿼리 차원(일반적으로 300)을 포함하는 텐서 탐색
            if 300 in xi.shape:
                if xi.shape[-1] == self.nc + 4:
                    pred_concat = xi
                elif xi.shape[-2] == self.nc + 4:
                    pred_concat = xi.transpose(-1, -2)
                elif xi.shape[-1] == 4:
                    pred_boxes = xi
                elif xi.shape[-1] == self.nc:
                    pred_scores = xi

        # 2. 바운딩 박스와 점수 추출
        if pred_concat is not None:
            boxes = pred_concat[0, :, :4]
            scores = pred_concat[0, :, 4:]
        elif pred_boxes is not None and pred_scores is not None:
            boxes = pred_boxes[0]
            scores = pred_scores[0]
        else:
            shapes = [xi.shape for xi in x]
            raise ValueError(f"RT-DETR 디코더 출력을 찾을 수 없습니다 (300 쿼리 예상). "
                             f"현재 텐서 모양들: {shapes}. 트랜스포머 디코더가 MXQ에 정상적으로 컴파일되었는지 확인하세요.")

        # 3. 박스 형태 변환: (cx, cy, w, h) -> (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        # 4. 각 쿼리에서 가장 높은 점수와 해당 클래스 라벨 가져오기
        max_scores, labels = torch.max(scores, dim=-1)

        # 5. 설정한 신뢰도(Confidence) 임계값으로 필터링
        mask = max_scores > self.conf_thres
        boxes_xyxy = boxes_xyxy[mask]
        max_scores = max_scores[mask]
        labels = labels[mask]

        if len(boxes_xyxy) == 0:
            return None

        # 6. Visualizer 요구 사항에 맞게 병합: [x1, y1, x2, y2, conf, cls] -> (N, 6) 형태
        result = torch.cat([boxes_xyxy, max_scores.unsqueeze(-1), labels.float().unsqueeze(-1)], dim=-1)

        return [result]