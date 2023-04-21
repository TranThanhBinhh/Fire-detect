import cv2
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh, increment_path
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from pathlib import Path
import os
import datetime
import torch
import requests
save_dir = 'result'
os.makedirs(save_dir, exist_ok=True)
hide_labels = False
hide_conf = False

url = 'http://localhost:8000/fire_alert/image'

def detect_camera(weights, conf_thres=0.5, iou_thres=0.45):
    device = select_device('')
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model.eval()

    imgsz = check_img_size(640, s=model.stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    cap = cv2.VideoCapture(0) 

    image = cv2.imread("blank_img.png", 0)
    cv2.imshow('YOLOv5 Object Detection', image)

    while True:
        ret, frame = cap.read()
        temp = frame
        if not ret:
            print('Không thể đọc khung hình từ camera')
            break

        img = cv2.resize(frame, (imgsz, imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img / 255.0  # data normalization
        img = torch.from_numpy(img).unsqueeze(0).to(device) 
        img = img.float()

        with torch.no_grad(): #detect fire
            pred = model(img, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres, iou_thres)

        time = str(datetime.datetime.now()).replace(':', '.')
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator = Annotator(frame, line_width=3, example=str(names))
                    annotator.box_label(xyxy, label, color=colors(c, True))
                #     save_one_box(xyxy,temp, file=Path(save_dir + '/' + f'{i}.jpg'), BGR=True)
                img_name = os.path.join(save_dir, time + '.jpg')  # Tên file lưu
                cv2.imwrite(img_name, frame)
                files = {'image': (img_name, open(img_name, 'rb'), "image/png"),}
                with requests.Session() as s:
                    s.post(url, files=files)        
        
        # cv2.imshow('YOLOv5 Object Detection', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
detect_camera('best.pt', 0.6, 0.6)
