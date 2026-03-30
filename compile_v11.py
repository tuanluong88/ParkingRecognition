from qubee import mxq_compile

if __name__ == "__main__":
    mxq_compile(
        model="./modified_best_yolo200_cut_sim.onnx", # 1단계에서 만든 수술된 모델
        save_path="./modified_best_yolo200.mxq",  # 최종 생성될 mxq 파일 이름
        calib_data_path="./calib_images/npy/yolov11.txt", # 2단계에서 만든 캘리브레이션 데이터 경로
        quantize_method="maxpercentile",
        quantize_percentile=0.9999,
        topk_ratio=0.01,
        device='cpu'  # 도커 환경에 맞춰 CPU로 변경
    )