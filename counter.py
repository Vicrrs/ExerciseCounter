import math

import cv2
import mediapipe as mp

video = cv2.VideoCapture(r'Videos/polichinelos.mp4')
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
count = 0
check = True

while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h, w, _ = img.shape

    if points:
        peDY = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
        peDX = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)
        peEY = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
        peEX = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)

        distMO = math.hypot(moDX-moEX, moDY-moEY)
        distPE = math.hypot(peDX-peEX, peDY-peEY)
        print(f'MAOS: {distMO}. PES: {distPE}')

        if check == True and distMO <= 150 and distPE >= 150:
            count += 1
            check = False

        if distMO > 150 and distPE < 150:
            check = True
        print(count)

        text = (f'Quantidade: {count}')
        cv2.rectangle(img, (20, 240), (280, 280), (255, 0, 0), -1)
        cv2.putText(img, text, (40, 265), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)

    cv2.imshow('videoRGB', img)
    cv2.waitKey(40)
