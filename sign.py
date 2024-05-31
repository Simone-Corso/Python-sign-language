# installar le librerie per importare i modelli

import mediapipe as mp
import cv2


# configuriamo istanza per rivelamento delle mani

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# catturiamo video utilizzando OpenCv

cap = cv2.VideoCapture(0)

# creo un loop per elaborare ogni frame del video.

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignorando il frame vuoto.")
        continue

    # Converti l'immagine da BGR a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Inverti l'immagine per un'esperienza selfie
    image = cv2.flip(image, 1)

    # Rendi l'immagine non scrivibile per migliorare le prestazioni
    image.flags.writeable = False

    # Rileva le mani
    results = hands.process(image)

    # Rendi l'immagine scrivibile e converti nuovamente a BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Controlla se sono state rilevate delle mani
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegna le annotazioni delle mani
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostra il frame elaborato
    cv2.imshow('Hand Gesture Recognition', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Questi punti di riferimento possono essere utilizzati per riconoscere gesti specifici

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(id, cx, cy)


