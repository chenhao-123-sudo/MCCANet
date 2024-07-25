# -*- coding: utf-8 -*-
from ultralytics import YOLO

#将训练好的网络模型导出为onnx格式
model = YOLO("runs/detect/train17/weights/best.pt")

path = model.export(format="onnx")  # export the model to ONNX format