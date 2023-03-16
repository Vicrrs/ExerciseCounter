import cv2
import mediapipe as mp


# Taking the path of the video
video = cv2.VideoCapture()

while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imshow('videoRGB', img)
    cv2.waitKey(0)