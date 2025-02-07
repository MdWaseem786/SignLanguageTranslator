import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp

model = tf.keras.models.load_model(r"D:\SignLanguageTranslator\Model\trained_model.h5")
labels = {0: "A",1: "F", 2: "L"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img = cv2.resize(hand_img, (128, 128))
            hand_img = np.expand_dims(hand_img, axis=0) / 255.0
            prediction = model.predict(hand_img)
            label = labels[np.argmax(prediction)]
            cv2.putText(frame, label, (x_min, y_min - 10), font, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Sign Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
