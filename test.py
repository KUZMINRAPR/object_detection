import torch
from ultralytics import YOLO
from pathlib import Path
import logging


model = YOLO('runs/detect/train3/weights/best.pt')

logging.basicConfig(
    level=logging.INFO,
    filename='logs/test.log',
)

input_dir = Path('dataset/images/val')

for img in input_dir.iterdir():
    results = model(img,save=True)
    logging.info(f'{img.name} {len(results[0])}')
