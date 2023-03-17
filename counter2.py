import math

import cv2
import mediapipe as mp

# Definindo o caminho do video
VIDEO_PATH = 'Videos/polichinelos.mp4'

# Iniciando o objeto de video
cap = cv2.VideoCapture(VIDEO_PATH)

# Config do modelo de pose
pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configurando a exibição dos landmarks
drawing = mp.solutions.drawing_utils

# Iniciando contagem e verificação
count = 0
verify = True


while True:
    # Lendo frame por frame do video
    sucess, image = cap.read()

    if not sucess:
        break

    # Convertendo imagem para o RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processando a imagem e extraindo o landmark
    results = pose.process(image_rgb)
    landmarks = results.pose_landmarks

    # Desenhando os pontos na imagem
    drawing.draw_landmarks(
        image, landmarks, mp.solutions.pose.POSE_CONNECTIONS)

    # if landmarks:
    #     print(landmarks)

    # Calculando as distancias entre as mãoes e os pés
    if landmarks is not None:
        h, w, _ = image.shape

    # Trabalhando para achar aquele momento de interesse
    right_foot_y = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
    right_foot_x = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)
    left_foot_y = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
    left_foot_x = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)
    right_hand_y = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].y*h)
    right_hand_x = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].x*w)
    left_hand_y = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX].y*h)
    left_hand_x = int(
        landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX].x*w)

    # Calculando a distancia entre as mãos e os pés
    distance_hand = math.hypot(
        right_hand_x - left_hand_x, right_hand_y - left_hand_y)
    distance_feet = math.hypot(
        right_foot_x - left_foot_x, right_foot_y - left_foot_y)

    # print(f"MÃOS: {distance_hand}, PÉS: {distance_feet}")

    # Verificar se tem alguma repetição no exercício
    if verify and distance_hand <= 150 and distance_feet >= 150:
        count += 1
        verify = False

    # Reiniciar a verificação
    if distance_hand > 150 and distance_feet < 150:
        verify = True

    # Exibindo a contagem na tela
    text = (f"Quantidade: {count}")
    # print(text)
    cv2.rectangle(image, (20, 240), (280, 280), (255, 0, 0), -1)
    cv2.putText(image, text, (40, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Video', image)
    cv2.waitKey(40)
