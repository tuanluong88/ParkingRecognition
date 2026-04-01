from ultralytics import YOLO

model_yolo=YOLO('yolo11n.pt')
DATAPATH='/home/tuanluong/data/data.yaml'
EPOCHS=1000
results_yolo=model_yolo.train(data=DATAPATH, epochs=EPOCHS, imgsz=640, 
                              batch=8, 
                              project='/home/tuanluong/coop2026_1/SKKUdatatraining', 
                              name='skku_yolo11_epoch1000', 
                              patience=0)
