import onnx
from onnx import shape_inference

# 우리가 잘라냈던 모델
input_path = "./modified_best.onnx"
# Shape 정보가 꽉 채워진 새로운 모델
inferred_path = "./modified_best_inferred.onnx"

print("🔍 ONNX 모델의 누락된 Shape 정보를 다시 추론합니다...")

# 모델 로드
model = onnx.load(input_path)

# Shape 추론(Inference) 실행 (에러의 원인이었던 빈 공간을 채움)
inferred_model = shape_inference.infer_shapes(model)

# 새 파일로 저장
onnx.save(inferred_model, inferred_path)

print(f"✅ 완료! 새 모델이 저장되었습니다: {inferred_path}")