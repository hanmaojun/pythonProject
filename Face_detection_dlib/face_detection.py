import dlib
import face_recognition
import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_image(image, title):
    image_rgb = image[:, :, ::-1]
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis("off")


def draw_result_face(image, faces, predictor):
    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 3)

        shape = predictor(image, face)
        for pt in shape.parts():
            pt_position = (pt.x, pt.y)
            cv2.circle(image, pt_position, 2, (0, 0, 255), -1)

    return image


def show_landmarks(image, landmarks):
    for landmark_dict in landmarks:
        for landmark_key in landmark_dict.keys():
            for point in landmark_dict[landmark_key]:
                cv2.circle(image, point, 3, (255, 0, 0), -1)

    return image


def main():
    # 加载图片
    img = cv2.imread("C:\\Users\\hanma\\PycharmProjects\\pythonProject\\images\\Japanese_Women_1.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载摄像头
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if cap.isOpened() is False:
        print("Camera Error!")

    fourcc = cv2.VideoWriter_fourcc(*"XIVD")
    output_camera = cv2.VideoWriter("video_output", fourcc, int(fps), (int(frame_width), int(frame_height)), False)

    # 调用检测模型
    face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "C:\\Users\\hanma\\PycharmProjects\\pythonProject\\shape_predictor_68_face_landmarks.dat")

    # 创建画布
    # figure = plt.figure(figsize=(16, 9))
    # plt.suptitle("Face Detection by Dlib", fontsize=14, fontweight="bold")

    # show_image(result, "detection result")

    # plt.show()

    # 输出摄像头
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_output = frame
            output_camera.write(frame_output)
            # 输出检测结果

            # Dlib landmarks Predictor
            landmarks = predictor.
            frame_output = show_landmarks(frame, landmarks)
            output_camera.write(frame_output)
            cv2.imshow("Frame", frame_output)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
