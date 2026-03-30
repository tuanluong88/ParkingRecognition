from ultralytics import RTDETR

#training rtdetr code
#epoch is too many=> you should set amp=False(in this case, epochs=200, so I set amp=False)
#lr0 is too big=> loss doesn't go down well, so I set lr0=0.0001
#to make code working well, I set warmup_bias_lr=0.0001
#optimizer=> if you don't set this, then this set to 'auto'. => your set lr0 does't apply => so I set this 'AdamW'
#momentum=0.9, lrf=0.01, cos_lr=False is just setting value that if you don't write it.

model_rtdetr=RTDETR('rtdetr-l.pt')
DATAPATH='/home/tuanluong/data/data.yaml'
EPOCHS=1000
results_rtdetr=model_rtdetr.train(data=DATAPATH, epochs=EPOCHS, 
                                  imgsz=640, 
                                  batch=8, 
                                  cos_lr=False, 
                                  optimizer='AdamW', 
                                  momentum=0.9, 
                                  amp=False, 
                                  lr0=0.0001, 
                                  lrf=0.01, 
                                  warmup_epochs=3, 
                                  warmup_bias_lr=0.0001, 
                                  project='/home/tuanluong/coop2026_1/SKKUdatatraining',
                                  name='skku_rtdetr_epoch1000')
