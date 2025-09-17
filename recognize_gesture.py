import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
import os
import pyttsx3
import threading

# Inizializza il motore di sintesi vocale e il recognizer
engine = pyttsx3.init()
r = sr.Recognizer()

# Inizializza Mediapipe e OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funzione per la sintesi vocale
def text_to_speech(text):
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Avvia la cattura video
cap = cv2.VideoCapture(0)

# Carica i dati dei gesti
folder_path = "sign"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
file_path = os.path.join(folder_path, 'gestures.npy')
if os.path.exists(file_path):
    gesti = np.load(file_path, allow_pickle=True).item()
else:
    gesti = {}

def riconosci_gesto(landmarks):
    for label, saved_landmarks in gesti.items():
        if np.allclose(saved_landmarks, landmarks, atol=0.05):
            return label
    return "Gesto sconosciuto"

# Variabile globale per il sottotitolo (accessibile da entrambi i thread)
sottotitolo = ""

# Funzione per il riconoscimento vocale che girerà in un thread separato
def speech_recognition_thread():
    global sottotitolo
    while True:
        try:
            with sr.Microphone() as source:
                audio = r.listen(source, timeout=0.5, phrase_time_limit=1)
                testo_riconosciuto = r.recognize_google(audio, language="it-IT")
                sottotitolo = testo_riconosciuto
        except (sr.UnknownValueError, sr.WaitTimeoutError):
            pass
        except Exception as e:
            sottotitolo = ""
        
# Creiamo e avviamo il thread del riconoscimento vocale
speech_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
speech_thread.start()

# Variabile per contare i frame e memorizzare il gesto
frame_counter = 0
riconosciuto = "Gesto sconosciuto"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Ridimensiona il frame per un'elaborazione più veloce
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    
    # Processa il riconoscimento solo ogni 5 frame per un'esperienza più fluida
    if frame_counter % 5 == 0:
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                nuovo_gesto = riconosci_gesto(landmarks)
                # Controlla se il gesto è cambiato prima di parlare
                if nuovo_gesto != riconosciuto and nuovo_gesto != "Gesto sconosciuto":
                    riconosciuto = nuovo_gesto
                    text_to_speech(riconosciuto)
        else:
            riconosciuto = "Gesto sconosciuto"

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Visualizza il testo del gesto sullo schermo
    cv2.putText(image, riconosciuto, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Visualizza il sottotitolo, che viene aggiornato in background dall'altro thread
    if sottotitolo:
        cv2.putText(image, sottotitolo, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Riconoscimento dei gesti', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
