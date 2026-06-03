import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import qbruntime
import onnxruntime as ort  # ⭐ 추가됨!

from postprocess import RTDETRPostProcess
from visualize import RTDETRVisualizer

def preprocess_yolo(img_path: str, img_size=(640, 640)):
    # 기존과 동일한 NPU 최적화 전처리 (NHWC, uint8 유지)
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
    
    img = np.ascontiguousarray(img)  
    img = np.expand_dims(img, axis=0) # Shape: (1, 640, 640, 3)
    
    return img

class HybridRTDETR_ORT:
    def __init__(self, mxq_path: str, decoder_onnx_path: str):
        print("[Hybrid] NPU 모델(Encoder)을 초기화합니다...")
        self.acc = qbruntime.Accelerator()
        self.mc = qbruntime.ModelConfig()
        self.mc.set_single_core_mode(None, [qbruntime.CoreId(qbruntime.Cluster.Cluster0, qbruntime.Core.Core0)])
        self.npu_model = qbruntime.Model(mxq_path, self.mc)
        self.npu_model.launch(self.acc)

        print(f"[Hybrid] ONNX Runtime(Decoder)을 {decoder_onnx_path}에서 초기화합니다...")
        # ⭐ PyTorch 대신 ONNX Runtime 세션 시작 (CPU 엔진 사용)
        self.ort_session = ort.InferenceSession(decoder_onnx_path, providers=['CPUExecutionProvider'])
        
        # ONNX 모델이 기대하는 입력 이름 리스트 (feat_80, feat_40, feat_20)
        self.ort_input_names = [inp.name for inp in self.ort_session.get_inputs()]

        print("[Hybrid] 최고속(ORT) 파이프라인 초기화 완료!")

    def __call__(self, img_array):
        # 1. NPU 인코더 추론 (출력은 Numpy 리스트)
        npu_outputs = self.npu_model.infer([img_array])

        # 2. 특징맵 포맷팅 (Numpy 단에서 초고속 처리)
        feats = []
        for out in npu_outputs:
            # NHWC -> NCHW 포맷 변경
            if out.shape[-1] == 256 and len(out.shape) == 4:
                out = np.transpose(out, (0, 3, 1, 2))
            # ONNX가 기대하는 float32로 명시적 변환
            feats.append(out.astype(np.float32))

        # 해상도 내림차순(80, 40, 20)으로 정렬
        feats.sort(key=lambda x: x.shape[2] * x.shape[3], reverse=True)
        
        # 3. ONNX Runtime 디코더 추론
        ort_inputs = {
            self.ort_input_names[0]: feats[0], # feat_80
            self.ort_input_names[1]: feats[1], # feat_40
            self.ort_input_names[2]: feats[2]  # feat_20
        }
        
        # preds는 리스트 안에 Numpy 배열이 담긴 형태로 나옵니다.
        preds = self.ort_session.run(None, ort_inputs)

        # 4. 후처리(postprocess.py)가 PyTorch 텐서를 기대하므로 마지막에만 텐서로 변환
        return torch.from_numpy(preds[0])

    def dispose(self):
        self.npu_model.dispose()


if __name__ == "__main__":
    parser = ArgumentParser(description="Run High-Speed Hybrid RT-DETR inference")
    parser.add_argument("--mxq-path", type=str, default="./skku_rtdetr_epoch1000_encoder.mxq")
    parser.add_argument("--decoder-path", type=str, default="./decoder_only.onnx") # pt 대신 onnx로 변경
    parser.add_argument("--image-path", type=str, default="./Example1.jpg")
    parser.add_argument("--output-path", type=str, default="./Result1.jpg")
    parser.add_argument("--conf-thres", type=float, default=0.25)

    args = parser.parse_args()

    # ⭐ 클래스 이름 변경 적용
    hybrid_model = HybridRTDETR_ORT(args.mxq_path, args.decoder_path)
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