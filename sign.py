# installar le librerie per importare i modelli

import mediapipe as mp
import cv2


# configuriamo istanza per rivelamento delle mani

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# catturiamo video utilizzando OpenCv

cap = cv2.VideoCapture(0)
