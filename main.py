import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_face(img):
    face_img = img.copy()

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_rects = face_cascade.detectMultiScale(face_img)

    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)

    return face_img


def main():
    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read(0)

        frame = detect_face(frame)

        cv2.imshow('Video face detect', frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
