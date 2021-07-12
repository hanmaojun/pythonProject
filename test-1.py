import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_img_origin(image, title, pos):

    plt.subplot(3, 3, pos)
    plt.title(title)
    plt.imshow(image)


def show_img_gray(image, title, pos):
    img_BGR = image[:, :, ::-1]

    plt.title("title")
    plt.subplot(2, 3, pos)
    plt.imshow(img_BGR)


def show_histogram(hist, title, pos, color):
    plt.title(title)
    plt.subplot(3, 3, pos)
    plt.xlabel("Bins")
    plt.ylabel("Pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


def show_channel(img_channel, title, pos, color):
    plt.title(title)
    plt.subplot(3, 3, pos)
    plt.xlabel("Bins")
    plt.ylabel("Pixels")
    plt.xlim([0, 256])
    plt.plot(img_channel, color=color)


def main():
    plt.figure(figsize=(16, 10))
    plt.suptitle("Image Histogram", fontsize=14, fontweight="bold")

    img = cv2.imread("images/autumn.jpg")
    img = img[:, :, ::-1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # hist_img = cv2.calcHist(img_gray, [0], None, [256], [0, 256])
    hist_B= cv2.calcHist(img, [1], None, [256], [0, 256])
    hist_G = cv2.calcHist(img, [2], None, [256], [0, 256])
    hist_R = cv2.calcHist(img, [3], None, [256], [0, 256])

    img_BGR = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


    # show_img_gray(img_BGR, "BGR image", 1)
    show_histogram(hist_B, "Blue channel histogram", 4, 'm')
    show_histogram(hist_G, "Green channel histogram", 5, 'm')
    show_histogram(hist_R, "Red channel histogram", 6, 'm')

    show_img_origin(img, 'Original Image', 1)

    plt.show()


if __name__ == '__main__':
    main()
