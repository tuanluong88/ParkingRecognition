import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn as nn
import qbruntime

from ultralytics import RTDETR
from postprocess import RTDETRPostProcess
from visualize import RTDETRVisualizer


def preprocess_yolo(img_path: str, img_size=(640, 640)):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]  
    r = min(img_size[0] / h0, img_size[1] / w0)  
    new_unpad = int(round(w0 * r)), int(round(h0 * r))

    if (w0, h0) != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    dh, dw = img_size[0] - new_unpad[1], img_size[1] - new_unpad[0]  
    dw /= 2  
    dh /= 2  
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  
    
    # ========================================================
    # ⭐ [수정 1] NPU 메모리 구조에 맞게 NHWC (640, 640, 3)를 강제합니다.
    # 절대 np.transpose를 쓰지 마세요! 이미지가 노이즈로 깨집니다.
    # ========================================================
    img = np.ascontiguousarray(img)  
    img = np.expand_dims(img, axis=0) # Shape: (1, 640, 640, 3), dtype: uint8 기본값
    
    return img


class HybridRTDETR:
    def __init__(self, mxq_path: str, pt_path: str):
        print("[Hybrid] NPU 모델(Encoder)을 초기화합니다...")
        self.acc = qbruntime.Accelerator()
        self.mc = qbruntime.ModelConfig()
        self.mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
        self.npu_model = qbruntime.Model(mxq_path, self.mc)
        self.npu_model.launch(self.acc)

        print(f"[Hybrid] PyTorch 모델(Decoder)을 {pt_path}에서 추출합니다...")
        self.device = torch.device("cpu")
        
        pt_model = RTDETR(pt_path).model.to(self.device).eval()
        self.decoder = pt_model.model[-1]

        for i in range(len(self.decoder.input_proj)):
            self.decoder.input_proj[i] = nn.Identity()

        print("[Hybrid] 초기화 완료!")

    def __call__(self, img_array):
        npu_outputs = self.npu_model.infer([img_array])

        # ========================================================
        # ⭐ [수정 2] NPU ➡️ PyTorch 방탄(Foolproof) 브릿지
        # NPU 출력이 NHWC인지 NCHW인지, 순서가 어떻게 꼬였는지 상관없이
        # 무조건 PyTorch가 원하는 형태로 재조립합니다.
        # ========================================================
        feats = []
        for out in npu_outputs:
            feat = torch.from_numpy(out).to(self.device)
            # 만약 채널(256)이 맨 뒤에 있는 NHWC 포맷이라면 NCHW로 뒤집기
            if feat.shape[-1] == 256 and len(feat.shape) == 4:
                feat = feat.permute(0, 3, 1, 2)
            feats.append(feat)

        # 공간 크기(H*W)를 기준으로 내림차순 정렬하여 무조건 [80, 40, 20] 순서가 되도록 보장
        feats.sort(key=lambda x: x.shape[2] * x.shape[3], reverse=True)
        
        decoder_inputs = feats

        with torch.no_grad():
            preds = self.decoder(decoder_inputs)

        return preds

    def dispose(self):
        self.npu_model.dispose()


if __name__ == "__main__":
    parser = ArgumentParser(description="Run Hybrid RT-DETR inference (NPU + CPU)")
    parser.add_argument("--mxq-path", type=str, default="./skku_rtdetr_epoch1000_encoder.mxq")
    parser.add_argument("--pt-path", type=str, default="./best.pt")
    parser.add_argument("--image-path", type=str, default="./Example1.jpg")
    parser.add_argument("--output-path", type=str, default="./Result1.jpg")
    parser.add_argument("--conf-thres", type=float, default=0.25)

    args = parser.parse_args()

    hybrid_model = HybridRTDETR(args.mxq_path, args.pt_path)
    postprocess = RTDETRPostProcess(args.conf_thres)
    visualizer = RTDETRVisualizer()

    img = preprocess_yolo(args.image_path)
    outputs = hybrid_model(img)
    
    if isinstance(outputs, tuple):
        outputs_list = list(outputs)
    elif isinstance(outputs, torch.Tensor):
        outputs_list = [outputs]
    else:
        outputs_list = outputs
        
    result = postprocess(outputs_list)

    output_path = args.output_path or os.path.join(os.path.dirname(args.image_path), "output.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if result is not None:
        visualizer.save(result, input_path=args.image_path, output_path=output_path)
        print(f"✅ 객체 탐지 성공! 결과가 저장되었습니다: {output_path}")
    else:
        print("탐지된 객체가 없습니다. 임계값(conf-thres)을 낮춰보세요.")
        
    hybrid_model.dispose()