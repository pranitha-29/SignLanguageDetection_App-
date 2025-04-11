import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from collections import Counter
import mediapipe as mp

# Load model
model = load_model('sign_language_model.h5')
class_labels = ['hello', 'help','i_love_you', 'no','please','stop' 'thanks', 'yes']

# Color map for labels
color_map = {
    "hello": (255, 0, 0),
    "i_love_you": (255, 20, 147),
    "no": (0, 0, 255),
    "thanks": (0, 255, 255),
    "yes": (0, 255, 0),
}

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
x, y, w, h = 100, 100, 300, 300
predictions = []
CONFIDENCE_THRESHOLD = 0.85

print("🖐️ Show your hand sign inside the green box...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    label = "No hand detected"
    show_prediction = False

    if result.multi_hand_landmarks:
        roi = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        roi_input = np.reshape(roi_normalized, (1, 64, 64, 3))

        pred = model.predict(roi_input, verbose=0)
        confidence = np.max(pred)
        label_index = np.argmax(pred)

        if confidence >= CONFIDENCE_THRESHOLD:
            pred_label = class_labels[label_index]
            predictions.append(pred_label)
            show_prediction = True
        else:
            predictions.append("...")

    else:
        predictions.append("...")

    if len(predictions) > 10:
        predictions.pop(0)

    most_common = Counter(predictions).most_common(1)[0][0]

    if show_prediction and most_common != "...":
        color = color_map.get(most_common, (255, 255, 255))
        text = f"{most_common} ({confidence:.2f})"
    else:
        color = (128, 128, 128)
        text = "Waiting for hand..."

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
