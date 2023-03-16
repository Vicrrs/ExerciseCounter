import cv2
import mediapipe as mp


# Taking the path of the video
video = cv2.VideoCapture()
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    cv2.imshow('videoRGB', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
