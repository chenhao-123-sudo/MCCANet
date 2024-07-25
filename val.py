# -*- coding: utf-8 -*-
from ultralytics import YOLO


model = YOLO("runs/detect/test/best.pt")  
model.val(data="ultralytics/datasets/VisDrone.yaml",imgsz=800,batch=1,device=0) 
