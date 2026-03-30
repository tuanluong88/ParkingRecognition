import onnx
from onnx import helper

# 1. 사용자님의 원본 모델 불러오기
onnx_model = onnx.load("modified_best_yolo200.onnx")

print("변경 전 출력 노드:", onnx_model.graph.output)
del onnx_model.graph.output[:]

# 2. 방금 정확하게 찾아주신 3개의 핵심 노드 이름 적용
output_list = [
    '/model.23/cv2.2/cv2.2.1/conv/Conv_output_0',
    '/model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv_output_0',
    '/model.23/cv2.2/cv2.2.2/Conv_output_0',
    '/model.23/cv3.2/cv3.2.1/cv3.2.1.0/conv/Conv_output_0',
    '/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv_output_0',
    '/model.23/cv3.2/cv3.2.2/Conv_output_0'
]

shape_info = onnx.shape_inference.infer_shapes(onnx_model)
for value_info in shape_info.graph.value_info:
    if value_info.name in output_list:
        onnx_model.graph.output.append(value_info)
   
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, "modified_best_yolo200_cut.onnx")
print("6개의 Conv 노드로 정밀 수술 완료!")