import cv2
import argparse
import matplotlib as plt
import numpy as np
import dlib

def draw_rect(image, faces):
    # face: (x,y), height, width
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), 255, 5)
    return image


cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Frame Width:{}".format(frame_width))
print("Frame Height:{}".format(frame_height))
print("FPS:{}".format(fps))

#调用分类器
face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_dlib = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "C:\\Users\\hanma\\PycharmProjects\\pythonProject\\shape_predictor_68_face_landmarks.dat")

if cap.isOpened() is False:
    print("Camera Error!")

fourcc = cv2.VideoWriter_fourcc(*"XIVD")
output_gray = cv2.VideoWriter("video_output", fourcc, int(fps), (int(frame_width), int(frame_height)), False)

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # face_alt2_detect = face_alt2.detectMultiScale(frame)
        face_dlib_detect = face_dlib(frame)
        output_face = draw_rect(frame, face_dlib_detect)
        output_gray.write(output_face)

        cv2.imshow("Frame", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
