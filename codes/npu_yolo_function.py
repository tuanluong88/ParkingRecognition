import cv2
import numpy as np
import torch
import torchvision
import qbruntime

# --- [내부 후처리 함수] ---
def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, dtype=torch.float32) + grid_cell_offset
        sy = torch.arange(end=h, dtype=torch.float32) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float32))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def flatten_and_cat(feats):
    tensors = []
    for x in feats:
        t = torch.from_numpy(x)
        t = t.view(1, t.shape[1], -1)
        tensors.append(t)
    return torch.cat(tensors, dim=2).transpose(1, 2)

def process_mask_upsample_cv2(protos, masks_in, bboxes, shape=(640, 640)):
    c, mh, mw = protos.shape 
    protos_flat = protos.reshape(c, -1)
    masks = (masks_in @ protos_flat).reshape(-1, mh, mw)
    
    n = masks.shape[0]
    out_masks = np.zeros((n, shape[0], shape[1]), dtype=np.float32)
    
    for i in range(n):
        resized = cv2.resize(masks[i], (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        out_masks[i] = (resized > 0.0).astype(np.float32)
        
        x1, y1, x2, y2 = map(int, bboxes[i])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(shape[1], x2), min(shape[0], y2)
        
        mask_crop = np.zeros_like(out_masks[i])
        mask_crop[y1:y2, x1:x2] = out_masks[i][y1:y2, x1:x2]
        out_masks[i] = mask_crop
        
    return out_masks

def scale_boxes(img1_shape, coords, img0_shape):
    """640x640 크기의 박스 좌표를 원본 이미지 크기에 맞게 복원하는 함수"""
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    
    coords_scaled = coords.copy()
    coords_scaled[:, [0, 2]] -= pad[0]
    coords_scaled[:, [1, 3]] -= pad[1]
    coords_scaled[:, :4] /= gain
    
    coords_scaled[:, [0, 2]] = np.clip(coords_scaled[:, [0, 2]], 0, img0_shape[1])
    coords_scaled[:, [1, 3]] = np.clip(coords_scaled[:, [1, 3]], 0, img0_shape[0])
    return coords_scaled

# --- [YOLO 스타일 결과 객체] ---
class NPUResult:
    def __init__(self, boxes, masks, orig_shape):
        self.boxes = boxes if boxes is not None else np.empty((0, 6))
        self.masks = masks 
        self.orig_shape = orig_shape 

# --- [통합 NPU 모델 클래스] ---
class NPUYoloModel:
    def __init__(self, model_path="./best.mxq", conf_thres=0.25, iou_thres=0.45):
        self.acc = qbruntime.Accelerator()
        mc = qbruntime.ModelConfig()
        mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
        self.model = qbruntime.Model(model_path, mc)
        self.model.launch(self.acc)
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = (640, 640)

    def _preprocess(self, source):
        if isinstance(source, str):
            img = cv2.imread(source, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"이미지 경로 오류: {source}")
        elif isinstance(source, np.ndarray):
            img = source.copy()
        else:
            raise TypeError("입력은 파일 경로(str) 또는 Numpy 배열이어야 합니다.")

        orig_shape = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h0, w0 = img.shape[:2]
        r = min(self.img_size[0] / h0, self.img_size[1] / w0)
        new_unpad = int(round(w0 * r)), int(round(h0 * r))
        if (w0, h0) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        dh, dw = self.img_size[0] - new_unpad[1], self.img_size[1] - new_unpad[0]
        top, bottom = int(round(dh // 2)), int(round(dh - dh // 2))
        left, right = int(round(dw // 2)), int(round(dw - dw // 2))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        
        img = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
        return img.astype(np.uint8), orig_shape

    def predict(self, source):
        img_tensor, orig_shape = self._preprocess(source)
        outputs = self.model.infer([img_tensor])

        conf_20, box_20, mc_20 = outputs[0], outputs[1], outputs[2]
        conf_40, box_40, mc_40 = outputs[3], outputs[4], outputs[5]
        conf_80, box_80, mc_80 = outputs[6], outputs[7], outputs[8]
        proto = outputs[9][0]

        all_conf = flatten_and_cat([conf_20, conf_40, conf_80]).sigmoid()
        all_box = flatten_and_cat([box_20, box_40, box_80])
        all_mc = flatten_and_cat([mc_20, mc_40, mc_80])

        feats = [torch.from_numpy(box_20), torch.from_numpy(box_40), torch.from_numpy(box_80)]
        anchors, strides = make_anchors(feats, [32, 16, 8])
        
        x1y1 = anchors - all_box[0, :, :2]
        x2y2 = anchors + all_box[0, :, 2:]
        bboxes = torch.cat((x1y1, x2y2), -1) * strides

        scores = all_conf[0, :, 0]
        mc = all_mc[0]

        valid_idx = scores > self.conf_thres
        bboxes = bboxes[valid_idx]
        scores = scores[valid_idx]
        mc = mc[valid_idx]

        if len(bboxes) == 0:
            return [NPUResult(None, None, orig_shape)]

        keep = torchvision.ops.nms(bboxes, scores, self.iou_thres)
        bboxes = bboxes[keep]
        scores = scores[keep]
        mc = mc[keep]

        if len(bboxes) == 0:
            return [NPUResult(None, None, orig_shape)]

        bboxes_np = bboxes.numpy()
        scores_np = scores.numpy()
        mc_np = mc.numpy()

        class_ids = np.zeros_like(scores_np)
        det_boxes = np.concatenate([bboxes_np, scores_np[:, np.newaxis], class_ids[:, np.newaxis]], axis=1)

        masks = process_mask_upsample_cv2(proto, mc_np, bboxes_np, shape=self.img_size)

        return [NPUResult(det_boxes, masks, orig_shape)]

    def close(self):
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'acc'): del self.acc

    def __del__(self):
        self.close()

# --- [일반 YOLO 스타일의 세그멘테이션 시각화 함수] ---
def visualize_segment(orig_frame, result, color=(0, 255, 0)):
    """
    검출된 객체 위에 일반 YOLO 세그멘테이션처럼 연하게 색을 칠하고 사각형과 점수를 표시
    - orig_frame: 원본 이미지 (Numpy array)
    - result: model.predict()의 결과물
    - color: 마스크 및 박스 색상 (기본값: 초록색 BGR -> 0, 255, 0)
    """
    orig_h, orig_w = orig_frame.shape[:2]
    vis_frame = orig_frame.copy()
    
    # 검출 결과가 없으면 원본 그대로
    if result.masks is None or len(result.masks) == 0:
        cv2.imshow("NPU YOLO Segment Check", vis_frame)
        return

    # 1. 원본 크기의 빈 마스크 캔버스 생성
    combined_mask_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    # 2. 모든 개별 마스크를 하나로 병합
    for i in range(len(result.masks)):
        mask_640 = (result.masks[i] * 255).astype(np.uint8)
        mask_orig = cv2.resize(mask_640, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        combined_mask_orig = cv2.bitwise_or(combined_mask_orig, mask_orig)
        
    # 3. 마스크 영역에 색상 입히기 (반투명 오버레이)
    color_overlay = np.zeros_like(vis_frame)
    color_overlay[combined_mask_orig > 0] = color
    # 원본 이미지와 7:3 비율로 섞어 연하게 채색
    vis_frame = cv2.addWeighted(vis_frame, 1.0, color_overlay, 0.3, 0)

    # 4. 원본 크기로 복원된 바운딩 박스(Bbox)와 신뢰도(Conf)
    boxes_640 = result.boxes[:, :4]
    confs = result.boxes[:, 4]
    boxes_orig = scale_boxes((640, 640), boxes_640, (orig_h, orig_w))
    
    for i, box in enumerate(boxes_orig):
        x1, y1, x2, y2 = map(int, box)
        conf = confs[i]
        
        # 사각형 테두리
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        # 텍스트 라벨 (car와 신뢰도 소수점 2자리)
        label = f"car {conf:.2f}"
        cv2.putText(vis_frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 5. 화면 표시
    cv2.imshow("NPU YOLO Segment Check", vis_frame)