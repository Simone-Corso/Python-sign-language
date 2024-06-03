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
#WARNING ---> PREMENDO IL TASTO A O B O C PER INDICARE QUALE VUOI SALVARE.
label = "A"  # Imposta l'etichetta iniziale per il gesto 

def save_gesture(landmarks, label, count):
    gestures[f"{label}_{count}"] = landmarks

print("Muovi il gesto davanti alla videocamera. Premi 's' per salvare l'immagine con l'etichetta corrente...")

count = 0
while cap.isOpened() and count < 1000:
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

            # Mostra l'immagine e l'etichetta corrente
            cv2.putText(image, f'Label: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Recognition', image)
    
    # Controlla i tasti premuti
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esci con ESC
        break
    elif key == ord('s'):  # Salva il gesto con l'etichetta corrente
        if results.multi_hand_landmarks:
            save_gesture(landmarks, label, count)
            count += 1
            print(f"Gesto salvato con etichetta {label} - Totale salvati: {count}")
    elif key == ord('a'):  # Cambia l'etichetta a "A"
        label = "A"
        print("Etichetta cambiata a 'A'")
    elif key == ord('b'):  # Cambia l'etichetta a "B"
        label = "B"
        print("Etichetta cambiata a 'B'")

    time.sleep(0.1)  # Attende 100 millisecondi tra ogni cattura

cap.release()
cv2.destroyAllWindows()

# Salva i dati dei gesti in un file
np.save('gestures.npy', gestures)
print("Gesti salvati in gestures.npy")
print(f"Totale immagini catturate: {count}")
