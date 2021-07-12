import cv2
import dlib
import numpy as np


def main():
    capture = cv2.VideoCapture(1)

    detector = dlib.get_frontal_face_detector()

    tractor = dlib.correlation_tracker()

    tracking_state = False

    frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"XIVD")
    output = cv2.VideoWriter("video_output", fourcc, int(fps), (int(frame_width), int(frame_height)), False)

    pass

if __name__ == '__main__':
    main()