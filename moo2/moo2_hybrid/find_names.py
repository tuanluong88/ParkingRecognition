import onnx

model_path = "./best.onnx"
print(f"[{model_path}] 모델을 분석합니다...\n")
model = onnx.load(model_path)

# 'enc' 나 'decoder' 근처에 있는 노드들의 출력을 찾습니다.
print("🔍 인코더/디코더 관련 텐서 이름 후보:")
for node in model.graph.node:
    # 노드 이름이나 출력 이름에 아래 키워드가 포함된 경우 출력
    if "enc_output" in node.name or "decoder" in node.name:
        print(f"노드 이름: {node.name}")
        for out in node.output:
            print(f"  👉 실제 출력 텐서 이름: '{out}'")
print("\n위 리스트에서 디코더(decoder)로 들어가기 직전의 최종 인코더 출력 텐서 이름 2개를 찾아보세요.")