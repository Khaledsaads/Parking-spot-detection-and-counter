import cv2
from utils import *
import numpy as np
from pathlib import Path
def calc_diff(img1, img2):
    return np.abs(np.mean(img1)- np.mean(img2))

binary = cv2.imread(str("mask_1920_1080.png"), 0)
_, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

connected_components = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S) 
spots = get_parking_spots_bboxes(connected_components)
spots_status = [None for j in spots]
diffs = [None for j in spots]
video_path = Path("parking_1920_1080.mp4")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"{video_path} not opening")
previous_frame = None
frame_nmr = -1
ret = True
step = 30
while ret:
    frame_nmr+=1
    ret, frame = cap.read()
    if ret:
        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_idx, spot in enumerate(spots):
                x1, y1, w, h = spot
                diffs[spot_idx] = calc_diff(frame[y1: y1+h, x1: x1+w, :], previous_frame[y1: y1+h, x1 :x1+w, :])

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            for spot_idx in arr_:
                spot = spots[spot_idx]
                x1, y1, w, h = spot
                spot_crop = frame[y1: y1+h, x1: x1+w, :]
                spot_status =empty_or_not_svc(spot_crop)
                spots_status[spot_idx] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_idx, spot in enumerate(spots):
            spot_status = spots_status[spot_idx]
            x1, y1, w, h = spots[spot_idx]

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Available spots: {sum(spots_status)}/ {len(spots_status)}",(100, 60),
         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()