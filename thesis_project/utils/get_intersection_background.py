import cv2
import os
import numpy as np

def get_background():
    cap = cv2.VideoCapture(os.getcwd() + "/thesis_project/utils/2.MOV")
    first_iter = True
    frames = []
    i = 0
    while True:
        ret_val, frame = cap.read()
        if frame is None or i > 350:
            break
        if i % 1 == 0:
            frames.append(frame)
        i += 1
        print(len(frames))
        #if first_iter: # first iteration of the while loop
        #    avg = np.float32(frame)
        #    first_iter = False
        #else:
        #    cv2.accumulateWeighted(frame, avg, 0.005)
    background_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    #result = cv2.convertScaleAbs(avg)
    #cv2.imshow("result", result)
    cv2.imwrite("averaged_frame.jpg", background_frame)

if __name__ == "__main__":
    get_background()

