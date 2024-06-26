import cv2

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Errore: Impossibile aprire la fotocamera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Errore: Impossibile ricevere il frame.")
        break

    cv2.imshow('Test della fotocamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
