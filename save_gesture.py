import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Inizializza Mediapipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Avvia la cattura video
cap = cv2.VideoCapture(1)

# Dizionario per memorizzare i punti di riferimento dei gesti
gestures = {}
label = "A"  # Imposta l'etichetta iniziale per il gesto / PREMENDO LA 'A' PER CAMBIARE LA LETTERA.
label= "B"
start_capture = False  # Flag per iniziare la cattura dei gesti

def save_gesture(landmarks, label, count):
    gestures[f"{label}_{count}"] = landmarks

print("Muovi il gesto davanti alla videocamera. Premi 'a' per selezionare l'etichetta e Invio per iniziare la memorizzazione...")

count = 0
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

            if start_capture:
                # Salva il gesto automaticamente
                save_gesture(landmarks, label, count)
                count += 1

    cv2.imshow('Hand Gesture Recognition', image)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esci con ESC
        break
    elif key == ord('a'):  # Cambia l'etichetta
        label = input("Inserisci l'etichetta desiderata: ").upper()
    elif key == 13:  # Invio per iniziare la memorizzazione
        start_capture = True
        print("Inizia la memorizzazione...")

    time.sleep(0.1)  # Attende 100 millisecondi tra ogni cattura

cap.release()
cv2.destroyAllWindows()

# Salva i dati dei gesti nella cartella specificata
folder_path = "sign"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, 'gestures.npy')
np.save(file_path, gestures)
print("Gesti salvati in", file_path)
print(f"Totale immagini catturate: {count}")
