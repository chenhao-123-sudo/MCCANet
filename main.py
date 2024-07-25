# -*- coding: utf-8 -*-
from ultralytics import YOLO

model = YOLO("ultralytics/models/v8/MCCA.yaml").load("./yolov8m.pt")
model.train(data="ultralytics/datasets/VisDrone.yaml",
            epochs=200, batch=2, imgsz=640, patience=50)

