import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition


def draw_rect(image, faces):
    # face: (x,y), height, width
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), ((x + w), (x + h)), 255, 10)
    return image


def show_image(image, title, pos):
    image = image[:, :, ::-1]
    plt.subplot(2, 2, pos)
    plt.title(title)
    plt.imshow(image)
    plt.axis("off")


def show_landmarks(image, landmarks):
    for landmark_dict in landmarks:
        for landmark_key in landmark_dict.keys():
            for idx, point in landmark_dict[landmark_key]:
                pos = (point[0, 0], point[0, 1])
                cv2.circle(image, point, 3, (255, 0, 0), -1)
                cv2.putText(image, str(idx+1), pos, "bold", 0.2, (187, 255, 255), 1, cv2.LINE_AA)

    return image


def main():
    # 创建画布
    plt.figure(figsize=(16, 9))
    plt.suptitle("Face Detection with Haarcascade", fontsize=14, fontweight="bold")

    # 载入图片

    img = cv2.imread("C:\\Users\\hanma\\PycharmProjects\\pythonProject\\images\\Japanese_Women_1.png")
    print(img.shape)

    #载入摄像头信息
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)



    # 调用分类器
    face_alt = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    face_alt_detect = face_alt.detectMultiScale(img)

    face_alt2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    face_alt2_detect = face_alt2.detectMultiScale(img)

    face_alttree = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")
    face_alttree_detect = face_alttree.detectMultiScale(img)

    face_landmarks = face_recognition.face_landmarks(img, None, "large")

    # 输出分类后结果
    # result = draw_rect(img.copy(), face_alt_detect)
    result = show_landmarks(img.copy(), face_landmarks)
    # 显示分类结果
    show_image(result, 'Detection Result', 1)
    plt.show()


if __name__ == '__main__':
    main()
