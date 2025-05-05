import cv2
import os
import numpy as np

def emotionImage(emotion):
    emoji_path = os.path.join('Emojis', emotion.lower() + '.jpeg')
    if os.path.exists(emoji_path):
        image = cv2.imread(emoji_path)
    else:
        print(f"Emoji no encontrado para emoción: {emotion}")
        image = np.zeros((480, 300, 3), dtype=np.uint8)
    return cv2.resize(image, (300, 480))  # Asegura que el tamaño coincida para hconcat

# ----------- Selección del modelo ----------
method = 'LBPH'  # Cambia aquí si quieres probar EigenFaces o FisherFaces

if method == 'EigenFaces':
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
elif method == 'FisherFaces':
    emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
elif method == 'LBPH':
    emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
else:
    raise ValueError("Método inválido. Usa 'EigenFaces', 'FisherFaces' o 'LBPH'.")

emotion_recognizer.read(f'modelo{method}.xml')
# -------------------------------------------

dataPath = r'C:\Users\Usuario\Documents\IA-emociones\Data'
imagePaths = os.listdir(dataPath)
print('Emociones detectables:', imagePaths)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        label, confidence = emotion_recognizer.predict(rostro)

        if label >= len(imagePaths):
            print("Etiqueta fuera de rango:", label)
            continue

        emotion_name = imagePaths[label]
        print(f"Detectado: {emotion_name}, Confianza: {confidence:.2f}")

        if (method == 'EigenFaces' and confidence < 5700) or \
           (method == 'FisherFaces' and confidence < 500) or \
           (method == 'LBPH' and confidence < 60):

            cv2.putText(frame, emotion_name, (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            emoji = emotionImage(emotion_name)
            nFrame = cv2.hconcat([frame, emoji])

        else:
            cv2.putText(frame, 'No identificado', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            nFrame = cv2.hconcat([frame, np.zeros((480, 300, 3), dtype=np.uint8)])

    cv2.imshow('nFrame', nFrame)
    k = cv2.waitKey(1)
    if k == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
