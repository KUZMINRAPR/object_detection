from ultralytics import YOLO
from pathlib import Path

model = YOLO('runs/detect/train3/weights/best.pt')

input_dir = Path('dataset/images/val')

for img in input_dir.iterdir():
    results = model(img,save=True)


