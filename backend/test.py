# from ultralytics import YOLO
# import cv2

# model = YOLO("data/yolo/detect.pt")

# img = cv2.imread("test.png")  # Use any image to test
# results = model(img)[0]

# print("Classes:", model.names)

# for box in results.boxes:
#     cls = int(box.cls[0])
#     print("Detected:", model.names[cls], "Conf:", float(box.conf[0]))

import torch
torch.load("data/yolo/detect.pt", map_location="cpu")
