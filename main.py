import cv2 
import numpy as np
from ultralytics import YOLO 
import imutils
from collections import defaultdict


model = YOLO("models/yolov8n.pt")

cap = cv2.VideoCapture("videos/video.mp4")

track_history = defaultdict(lambda: [])

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)

    if ret == False:
        break 

    results = model.track(frame, persist=True, verbose = False)

    # comment -> ctrl + k + c
    # uncomment -> ctrl + k + u
    # print("results: ",results)     
    # print("results[0]: ",results[0])

    boxes = results[0].boxes.xywh.cpu()
    track_ids = results[0].boxes.id.int().cpu().tolist()

    annotated_frame = results[0].plot()

    

    for box, track_id in zip(boxes, track_ids):
        x,y,w,h = box
        track = track_history[track_id]
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        track.append((cx,cy))

        if len(track) > 30:
            track.pop(0)

        
        points = np.hstack(track).astype(np.int32).reshape((-1,1,2))
        cv2.polylines(annotated_frame,[points],isClosed= False, color = (0,255,0), thickness=3 )



    cv2.imshow("annotated_frame", annotated_frame)

    if cv2.waitKey(10) & 0XFF == ord("q"):
        break 
    

cap.release()
cv2.destroyAllWindows()