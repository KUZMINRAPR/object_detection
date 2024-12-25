from ultralytics import YOLO

model = YOLO('runs/detect/train3/weights/best.pt')

results = model('test1.jpg')

for result in results:
    result.show()