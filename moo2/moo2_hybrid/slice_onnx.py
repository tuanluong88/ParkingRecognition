import onnx

# 파일 경로 설정
input_path = "./best.onnx"
output_path = "./best_encoder_only.onnx"

# 1. 모델의 입력 이름 (로그에서 확인된 이름: images)
input_names = ["images"]

# 2. 새로운 출력 이름 지정 (로그에서 확인한 인코더 최종 출력 노드 이름)
# 로그의 "Output Name" 값을 그대로 가져옵니다.
output_names = [
    "/model.28/enc_output/enc_output.0/Add_output_0/reshape/layernorm/conv2d/reducemax",
    "/model.28/enc_output/enc_output.0/Add_output_0/reshape/layernorm"
]

print("ONNX 모델을 분리하는 중입니다...")
# 3. 모델 잘라내기
onnx.utils.extract_model(input_path, output_path, input_names, output_names)
print(f"성공적으로 분리되었습니다! 저장 위치: {output_path}")