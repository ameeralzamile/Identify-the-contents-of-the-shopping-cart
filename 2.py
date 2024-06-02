import cv2
import numpy as np  # Add this import for NumP5

img = cv2.imread('images/elephant.jpg')
classnames = []  # ليست أو array
classifile = 'files/thing.names'

with open(classifile, 'rt') as f:
    classnames = f.read().rstrip('\n').split('\n')
p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

net = cv2.dnn_DetectionModel(p, v)  # الكشف والفحص
net.setInputSize(320, 230)  # العرض والارتفاع
net.setInputScale(1.0 / 127.5)  # القياس
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)  # نظام الألوان

classIds, confs, bbox = net.detect(img, confThreshold=0.1)

# Convert tuples to NumPy arrays
classIds = np.array(classIds)
confs = np.array(confs)
bbox = np.array(bbox)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    cv2.putText(img, classnames[classId - 1],
                (box[0] + 10, box[1] + 20),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=2)

cv2.imshow('Identify the contents of the shopping cart', img)
cv2.waitKey(0)
