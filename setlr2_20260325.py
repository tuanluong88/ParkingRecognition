from ultralytics import RTDETR

model_rtdetr=RTDETR('rtdetr-l.pt')
DATAPATH='/home/tuanluong/coop2026_1/ParkingLot/data.yaml'
resutls_rtdetr=model_rtdetr(data=DATAPATH, epochs=200, imgsz=640, batch=8, cos_lr=False, optimizer='AdamW', momentum=0.9, amp=False, lr0=0.0001, lrf=0.01, warmup_epochs=3, warmup_bias_lr=0.0001, name='rtdetr_epoch200_setlr2')
