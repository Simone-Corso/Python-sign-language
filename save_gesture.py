import cv2
import mediapipe as mp
import numpy as np
import time

# Inizializza Mediapipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Avvia la cattura video
cap = cv2.VideoCapture(1)

# Dizionario per memorizzare i punti di riferimento dei gesti
gestures = {}
label = "A"  # Imposta l'etichetta desiderata per il gesto
label = "B"
def save_gesture(landmarks, label, count):
    gestures[f"{label}_{count}"] = landmarks

print("Muovi il gesto davanti alla videocamera. Cattura automatica di 100 immagini in corso...")

count = 0
while cap.isOpened() and count < 100: #scatta il momento fino a 1000 volte ( test )
                                      #(l'ideale per avere una perfomance giusta, sarebbe quella di 1000)
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

            # Salva il gesto automaticamente
            save_gesture(landmarks, label, count)
            count += 1

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

time.sleep(0.1)  # Attende ( prova di 1000 millisecondi tra ogni cattura) - test.

cap.release()
cv2.destroyAllWindows()

# Salva i dati dei gesti in un file
np.save('gestures.npy', gestures)
print("Gesti salvati in gestures.npy")
print(f"Totale immagini catturate: {count}")
