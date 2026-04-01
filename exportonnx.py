from ultralytics import RTDETR
import onnx
from onnx import shape_inference # specify the node names

model = RTDETR('/home/rise/Sim/miniconda3/envs/conv_env/convert/best_yolo1000.pt')
model.export(
format='onnx',
dynamic=False,
simplify=False,
opset=16,
imgsz=640,
nms=False,
)

onnx_path = '/home/rise/Sim/miniconda3/envs/conv_env/convert/best_yolo1000.onnx'
model_onnx = onnx.load(onnx_path)
model_inferred = shape_inference.infer_shapes(model_onnx)
onnx.save(model_inferred, onnx_path)
print(f"Shape inference 완료: {onnx_path}")