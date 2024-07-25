from ultralytics import YOLO

# model = YOLO("ultralytics/models/v8/yolov8.yaml").load("./yolov8m.pt")
# model = YOLO("ultralytics/models/v8/MFRENet.yaml").load("/home/user/chenhao/ultralytics/yolov8m.pt")
model = YOLO("runs/detect/train/weights/last.pt")

model.train(data="ultralytics/datasets/VisDrone.yaml",
            resume=True)
