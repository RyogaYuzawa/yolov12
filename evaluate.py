from ultralytics import YOLO

# model = YOLO('yolov12n.pt')
model = YOLO('best.pt')
model.val(data='coco.yaml', save_json=True)