from ultralytics import RTDETR

model_rtdetr=RTDETR('/home/tuanluong/coop2026_1/runs/detect/rtdetr_epoch200_lrgrad2/weights/last.pt')
results_rtdetr=model_rtdetr.train(resume=True)
