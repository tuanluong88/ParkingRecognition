import torch
import torch.nn as nn
from ultralytics import RTDETR

# 1. 모델 로드 및 디코더 추출
pt_path = "./best.pt"
print(f"'{pt_path}'에서 모델을 불러옵니다...")
pt_model = RTDETR(pt_path).model.eval()
decoder = pt_model.model[-1]

# 2. ONNX 변환을 위한 래퍼(Wrapper) 클래스 생성
# ONNX는 리스트 입력을 처리하기 까다로우므로, 3개의 텐서를 개별적으로 받도록 감싸줍니다.
class DecoderWrapper(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.decoder = dec
        # NPU가 이미 처리한 input_proj 중복 연산 방지 (이전에 했던 작업)
        for i in range(len(self.decoder.input_proj)):
            self.decoder.input_proj[i] = nn.Identity()

    def forward(self, feat_80, feat_40, feat_20):
        # 파이토치 디코더가 기대하는 리스트 형태로 묶어서 전달
        return self.decoder([feat_80, feat_40, feat_20])

wrapper = DecoderWrapper(decoder).eval()

# 3. 더미(Dummy) 입력 텐서 생성 (입력 해상도 640x640, 채널 256 기준)
# NPU에서 출력되어 디코더로 들어갈 3개의 피처맵 크기를 시뮬레이션합니다.
dummy_80 = torch.randn(1, 256, 80, 80)
dummy_40 = torch.randn(1, 256, 40, 40)
dummy_20 = torch.randn(1, 256, 20, 20)

output_onnx_path = "./decoder_only.onnx"
print("ONNX 변환을 시작합니다. (시간이 조금 걸릴 수 있습니다)...")

# 4. ONNX 추출
torch.onnx.export(
    wrapper,
    (dummy_80, dummy_40, dummy_20),     # 모델 입력
    output_onnx_path,                   # 저장 경로
    input_names=['feat_80', 'feat_40', 'feat_20'], # 입력 노드 이름
    output_names=['preds'],             # 출력 노드 이름
    opset_version=16,                   # RT-DETR에 안정적인 버전
    do_constant_folding=True            # 구조 최적화 적용
)

print(f"✅ 디코더 ONNX 추출 완료! 저장 위치: {output_onnx_path}")