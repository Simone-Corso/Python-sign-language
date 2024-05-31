import cv2
import mediapipe as mp
import numpy as np

# Inizializza Mediapipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Avvia la cattura video
cap = cv2.VideoCapture(1)

# Dizionario per memorizzare i punti di riferimento dei gesti
gestures = {}

def save_gesture(landmarks, label):
    gestures[label] = landmarks

print("Premi 's' per salvare il gesto corrente, seguito dalla lettera del gesto. Premi 'Esc' per uscire.")

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

            # Mostra i punti di riferimento sullo schermo
            cv2.putText(image, "Premi 's' per salvare il gesto", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Salva il gesto quando viene premuto il tasto 's'
            if cv2.waitKey(1) & 0xFF == ord('s'):
                label = input("Inserisci la lettera per il gesto corrente: ")
                save_gesture(landmarks, label)
                print(f"Gesto '{label}' salvato.")

    cv2.imshow('Hand Gesture Recognition', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Salva i dati dei gesti in un file
np.save('gestures.npy', gestures)
print("Gesti salvati in gestures.npy")
