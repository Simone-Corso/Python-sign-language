import cv2
import mediapipe as mp
import numpy as np
import os

# Inizializza Mediapipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Avvia la cattura video
cap = cv2.VideoCapture(0)

# Carica i dati dei gesti
folder_path = "sign"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, 'gesturesA.npy')  # Modifica il nome del file se necessario
if os.path.exists(file_path):
    gesti = np.load(file_path, allow_pickle=True).item()
else:
    gesti = {}  # Inizializza un dizionario vuoto se il file non esiste

def riconosci_gesto(landmarks):
    for label, saved_landmarks in gesti.items():
        if np.allclose(saved_landmarks, landmarks, atol=0.05):  # Tolleranza per il confronto
            return label
    return "Gesto sconosciuto"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            gesto = riconosci_gesto(landmarks)
            cv2.putText(image, gesto, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Riconoscimento dei gesti', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
