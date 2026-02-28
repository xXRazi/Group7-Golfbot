import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8s.pt')
while True:
    results = model.predict(source=0,show=True,stream=True)
    for result in results:
        annotated_frame = result.plot()
    cv.imshow('predict', annotated_frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break
