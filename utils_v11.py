import cv2
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torchvision

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    ori_shape=img.shape
    h0, w0 = img.shape[:2]  # orig hw
    r = min(img_size[0] / h0, img_size[1] / w0 ) # ratio
    if r != 1:  # if sizes are not equal
        h, w = (min(math.ceil(h0 * r), img_size[0]), min(math.ceil(w0 * r), img_size[1]))
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    r = min(r, 1.0)
    new_unpad = int(round(img.shape[1] * r)), int(round(img.shape[0] * r))
    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if (w0, h0) != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(114, 114, 114))  # add border
    img = (img / 255).astype(np.float32)
    
    return img

class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data = nn.Parameter(x.view(1, c1, 1, 1)).to(DEVICE)
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        x = x.to(DEVICE)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

def make_anchors(imh, imw, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    for stride in strides:
        h, w = imh // stride, imw // stride
        sx = torch.arange(end=w, dtype=torch.float32).to(DEVICE) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, dtype=torch.float32).to(DEVICE) + grid_cell_offset  # shift y
        sy, sx = meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32).to(DEVICE))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert x.shape[-1] == 4, f'input shape last dimension expected 4 but input shape is {x.shape}'
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

class YoloPostProcess():
    """YOLO object detection postprocess for anchorless models.
    """
    def __init__(self):
        self.is_first = True
        self.imh = self.imw = 640
        self.nc = 80
        self.conf_thres = 0.15
        self.iou_thres = 0.5
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = self.nc + self.reg_max * 4  # number of outputs per anchor
        self.dfl = DFL(self.reg_max)
        
        self.nl = 3
        self.stride = [2**(3+i) for i in range(self.nl)]
        self.anchors, self.strides = (xx.transpose(0, 1) for xx in make_anchors(self.imh, self.imw, self.stride, 0.5))
    
    def run(self, x):
        # mxq, torch inference without rearrange, decoding, concat
        # e.g. [(1, 144, 80, 80), (1, 144, 40, 40), (1, 144, 20, 20)]
        x = self.rearrange_npu_out(x)

        # --- [추가된 부분] 텐서 채널 수를 기반으로 nc 자동 추론 ---
        if self.is_first:
            # x[0]은 (Channels, H, W) 형태이므로 shape[0]이 총 채널 수
            channels = x[0].shape[0]
            # 전체 채널(66) - 박스 채널(reg_max*4 = 64) = 클래스 개수(2)
            self.nc = channels - (self.reg_max * 4) 
            self.no = channels
            self.is_first = False
        # -------------------------------------------------------------

        x = self.decode(x)
        
        x = self.nms(x)
        if x[0].nelement() == 0:
            return None
        return x
    
    def rearrange_npu_out(self, x):
        """ Reshape(from 1D to 4D) and NPU output
        
        Args:
            x: NPU output
        Returns:
            y: reshaped and transposed NPU output
        """

        y = []
        for i in range(self.nl):
            if isinstance(x[i], np.ndarray):
                tmp = torch.from_numpy(x[i]).to(DEVICE)
            elif isinstance(x[i], torch.Tensor):
                tmp = x[i].to(DEVICE)
            else:
                raise TypeError(f"Got unsupported type for input: {type(x[i])}.")
            
            # y.append(tmp.permute(0, 3, 1, 2).contiguous().detach().clone())
            y.append(tmp[0].contiguous().detach().clone())
        return y

    def decode(self, x):
        box, cls = torch.cat([xi.flatten(start_dim=1) for xi in x], 1).split((self.reg_max * 4, self.nc), 0)
        dbox = dist2bbox(self.dfl(box.unsqueeze(0)), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid().unsqueeze(0)), 1)
        return y
    
    def nms(
            self,
            prediction,
            classes=None,
            agnostic=False,
            labels=(),
            max_det=300,
            max_time_img=0.05,
            max_nms=30000,
            max_wh=7680,
    ):
        """
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L156
        Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

        Arguments:
            prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                containing the predicted boxes, classes, and masks. The tensor should be in the format
                output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                Valid values are between 0.0 and 1.0.
            classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
            agnostic (bool): If True, the model is agnostic to the number of classes, and all
                classes will be considered as one.
            labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
                list contains the apriori labels for a given image. The list should be in the format
                output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
            max_det (int): The maximum number of boxes to keep after NMS.
            max_time_img (float): The maximum time (seconds) for processing one image.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
                (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
        """

        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        bs = prediction.shape[0]  # batch size
        mi = 4 + self.nc  # mask start index (4 boxes, 80 classes)
        nm = prediction.shape[1] - mi # number of masks
        xc = prediction[:, 4:mi].amax(1) > self.conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        time_limit = 0.5 + max_time_img * bs  # seconds to quit after
        multi_label = self.nc > 1  # multiple labels per box (adds 0.5ms/img)
        
        prediction = prediction.transpose(-1,-2) # shape (1, 84, 8400) -> (1, 8400, 84)
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        
        t = time.time()
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), self.nc + nm + 5), device=x.device)
                v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
                v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Detections matrix nx6 (xyxy, cls, mask)
            box, cls, mask = x.split((4, self.nc, nm), 1)

            if multi_label:
                i, j = (cls > self.conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = cls.max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > self.conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            output[xi] = x[i]
            
            if (time.time() - t) > time_limit:
                print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output
    

if __name__=="__main__":
    img = preprocess_yolo("coco_sample/img/000000003538.jpg")
    print(img.shape)