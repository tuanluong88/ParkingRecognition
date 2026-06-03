import torch
from ultralytics import RTDETR

# 1. 모델 로드
print("모델을 로드하는 중...")
model = RTDETR("best.pt")
# 실제 PyTorch 모델(nn.Module)을 꺼내고 평가 모드(eval)로 설정합니다.
pt_model = model.model.eval() 

# 2. 디코더(Head) 레이어 확보
decoder = pt_model.model[-1]

# 3. Hook 함수 정의 (디코더에 데이터가 들어가기 직전에 실행됨)
def print_decoder_input_shape(module, inputs):
    print("\n" + "="*50)
    print("🔍 [디코더 입력 데이터 분석]")
    print("="*50)
    
    # inputs는 항상 튜플 형태의 위치 인자(Positional args)로 들어옵니다.
    # 일반적으로 첫 번째 원소(inputs[0])가 우리가 찾는 실제 입력 피처맵 데이터입니다.
    real_input = inputs[0] 
    print(f"입력 데이터의 최상위 타입: {type(real_input)}")
    
    # 리스트나 튜플로 여러 개의 텐서가 묶여서 들어오는 경우
    if isinstance(real_input, (list, tuple)):
        print(f"아이템 개수: {len(real_input)}개")
        for i, item in enumerate(real_input):
            if isinstance(item, torch.Tensor):
                print(f"  ▶ [{i}] Tensor: Shape {list(item.shape)}, Dtype: {item.dtype}")
            else:
                print(f"  ▶ [{i}] Other: Type {type(item)}")
                
    # 단일 텐서로 들어오는 경우
    elif isinstance(real_input, torch.Tensor):
        print(f"  ▶ 단일 Tensor: Shape {list(real_input.shape)}, Dtype: {real_input.dtype}")
    
    else:
        print(f"  ▶ 기타 타입: {type(real_input)}")
        
    print("="*50 + "\n")

# 4. 디코더에 Hook 등록
hook_handle = decoder.register_forward_pre_hook(print_decoder_input_shape)

# 5. 더미 이미지 생성 (배치1, 채널3, 높이640, 너비640)
# RT-DETR의 일반적인 입력 규격에 맞춥니다.
dummy_image = torch.randn(1, 3, 640, 640)

# 6. 추론 실행 
# 이 코드가 실행되면서 내부적으로 디코더를 지날 때 위의 Hook 함수가 자동 호출됩니다.
print("더미 이미지를 모델에 통과시킵니다...")
with torch.no_grad():
    _ = pt_model(dummy_image)

# 7. Hook 제거 (메모리 누수 방지용 정리)
hook_handle.remove()