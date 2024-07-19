from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.predict("bus.jpg", save=True, imgsz=640, conf=0.5, iou=0.7)