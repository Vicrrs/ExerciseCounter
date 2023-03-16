import cv2
import mediapipe as mp


# Taking the path of the video
video = cv2.VideoCapture(r'Videos/ANEXO_polichinelos.mp4')
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h, w, _ = img.shape

    if points:
        peDY = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].y*h)
        peDX = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].x*w)
        peEY = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].y*h)
        peEX = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].x*w)
        moDY = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].y*h)
        moDX = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].x*w)
        moEY = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].y*h)
        moEX = int(points.landmarks[pose.PoseLandmarks.RIGHT_FOOT_INDEX].x*w)

    cv2.imshow('videoRGB', img)
    cv2.waitKey(40)
