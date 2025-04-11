import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model # type: ignore
import threading
import time
import mediapipe as mp

# Load model and labels
model = load_model("best_model.h5")
class_labels = ['hello', 'help', 'i_love_you', 'no', 'please', 'stop', 'thanks', 'yes']

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

# GUI setup
root = tk.Tk()
root.title("🖐️ Sign Language Recognition")
root.geometry("750x600")
root.configure(bg="white")

lmain = tk.Label(root)
lmain.pack()

pred_label = tk.Label(root, text="Press Start to Begin", font=("Arial", 20), bg="white")
pred_label.pack(pady=15)

cap = cv2.VideoCapture(0)
recognizing = False
prev_prediction = ""
CONFIDENCE_THRESHOLD = 0.80


def predict_sign(roi):
    global prev_prediction
    roi_resized = cv2.resize(roi, (64, 64))
    roi_blurred = cv2.GaussianBlur(roi_resized, (3, 3), 0)
    roi_normalized = roi_blurred / 255.0
    roi_input = np.reshape(roi_normalized, (1, 64, 64, 3))

    pred = model.predict(roi_input, verbose=0)
    confidence = np.max(pred)
    label_index = np.argmax(pred)

    if confidence >= CONFIDENCE_THRESHOLD:
        prediction = class_labels[label_index]
        if prediction != prev_prediction:
            display_text = f"{prediction.replace('_', ' ')}"
            pred_label.config(text=display_text)
            prev_prediction = prediction


def recognize():
    global recognizing
    if not recognizing:
        return

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

        x_min = max(min([x for x, y in coords]) - 20, 0)
        y_min = max(min([y for x, y in coords]) - 20, 0)
        x_max = min(max([x for x, y in coords]) + 20, w)
        y_max = min(max([y for x, y in coords]) + 20, h)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size > 0:
            threading.Thread(target=predict_sign, args=(roi.copy(),), daemon=True).start()

        # Draw landmarks and bounding box
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        pred_label.config(text="Show a clear hand sign inside the box")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, recognize)


def start_recognition():
    global recognizing, prev_prediction
    recognizing = True
    prev_prediction = ""
    pred_label.config(text="Show a sign...")
    recognize()


def stop_recognition():
    global recognizing
    recognizing = False
    pred_label.config(text="Recognition Stopped")


# Buttons
btn_frame = tk.Frame(root, bg="white")
btn_frame.pack(pady=10)

start_btn = tk.Button(btn_frame, text="▶ Start", command=start_recognition,
                      font=("Arial", 16), bg="#4CAF50", fg="white", width=10)
start_btn.grid(row=0, column=0, padx=20)

stop_btn = tk.Button(btn_frame, text="■ Stop", command=stop_recognition,
                     font=("Arial", 16), bg="#F44336", fg="white", width=10)
stop_btn.grid(row=0, column=1, padx=20)


def on_closing():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
