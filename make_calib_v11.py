from qubee.calibration import make_calib_man
from utils_v11 import preprocess_yolo

make_calib_man(
    pre_ftn=preprocess_yolo,
    data_dir="./calib_images/img/",
    save_dir="./calib_images/npy/",
    save_name="yolov11",
    max_size=100
    )
