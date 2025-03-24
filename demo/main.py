import cv2
import numpy as np
from scipy.signal import butter
from scipy import signal
from tqdm import tqdm
from scipy import stats as st

from roi import getROI

def run():
    cap = cv2.VideoCapture("http://192.168.29.6:4747/video")
    
    if not cap.isOpened():
        print("Camera not found")
        return
    
    print("recoding..")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for _ in range(256):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (int(512 * frame.shape[1] / frame.shape[0]), 512))

        frame = getROI(frame)

        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

run()
        
# import cv2

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# print("Press 'q' to exit.")
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame. Exiting.")
#         break

#     # Resize while maintaining aspect ratio
#     height, width = frame.shape[:2]
#     aspect_ratio = width / height
#     new_width = int(512 * aspect_ratio)
#     resized_frame = cv2.resize(frame, (new_width, 512))

#     cv2.imshow('Video Capture', resized_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
