import sys

import numpy as np

import onnxruntime as ort

from utils_v11 import preprocess_yolo, YoloPostProcess
from visualize_v11 import YoloVisualizer

YOLO_ONNX_PATH = '/home/rise/Sim/yolov11/modified_best_yolo1000.onnx'
DATASET = [
    '/home/rise/Sim/yolov11/public03.jpg',
]

def main(args):
    model = ort.InferenceSession(YOLO_ONNX_PATH, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    postprocess = YoloPostProcess()
    visualizer = YoloVisualizer()
    
    for file_path in DATASET:
        file_name = file_path.split("/")[-1]
        img = preprocess_yolo(file_path)
        img = np.expand_dims(np.transpose(img, [2, 0, 1]), 0)
        outputs = model.run(None, {'images': img})
        result = postprocess.run(outputs)
        visualizer.save(result, input_path=file_path, output_path=f"output/{file_name}")
        

if __name__ == '__main__':
    main(sys.argv)
