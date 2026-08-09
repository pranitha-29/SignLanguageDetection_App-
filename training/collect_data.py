from pathlib import Path

import cv2
import mediapipe as mp


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

label = "please"  # Change this for the sign you want to collect
save_path = DATASET_DIR / label

save_path.mkdir(parents=True, exist_ok=True)

existing_images = len(
    [file for file in save_path.iterdir() if file.is_file()]
)

start_num = existing_images
num_new_images = 50
count = 0
IMG_SIZE = 64


# ============================================================
# Setup Camera
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# ============================================================
# MediaPipe Hands
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=0
)

mp_draw = mp.solutions.drawing_utils


# ============================================================
# Image Collection Settings
# ============================================================

frame_skip = 2
frame_count = 0

print(f"📸 Ready to collect '{label}' images")
print("▶️ Press 'c' to capture image when ONE hand is detected")
print("⎋ Press 'ESC' to stop")


# ============================================================
# Capture Images
# ============================================================

while count < num_new_images:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    frame_count += 1

    if frame_count % frame_skip != 0:
        continue

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)


    # ========================================================
    # Hand Detected
    # ========================================================

    if results.multi_hand_landmarks:

        h, w, _ = frame.shape

        hand_landmarks = results.multi_hand_landmarks[0]

        coords = [
            (
                int(lm.x * w),
                int(lm.y * h)
            )
            for lm in hand_landmarks.landmark
        ]


        # Calculate bounding box

        x_min = max(
            min(x for x, y in coords) - 20,
            0
        )

        y_min = max(
            min(y for x, y in coords) - 20,
            0
        )

        x_max = min(
            max(x for x, y in coords) + 20,
            w
        )

        y_max = min(
            max(y for x, y in coords) + 20,
            h
        )


        # Extract hand region

        roi = frame[
            y_min:y_max,
            x_min:x_max
        ]

        if roi.size == 0:
            continue


        # Resize image

        roi_resized = cv2.resize(
            roi,
            (IMG_SIZE, IMG_SIZE)
        )


        # ====================================================
        # Display Detection Information
        # ====================================================

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} ({count + 1}/{num_new_images})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Capture Image
        # ====================================================

        key = cv2.waitKey(1)

        if key == ord("c"):

            image_number = start_num + count

            image_path = (
                save_path /
                f"{label}_{image_number}.jpg"
            )

            cv2.imwrite(
                str(image_path),
                roi_resized
            )

            print(
                f"✅ Captured: {image_path.name}"
            )

            count += 1

        elif key == 27:  # ESC
            break


    # ========================================================
    # No Hand Detected
    # ========================================================

    else:

        cv2.putText(
            frame,
            "Show ONE hand clearly",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "One-Hand Sign Collector (Press 'c')",
        frame
    )


# ============================================================
# Cleanup
# ============================================================

cap.release()
hands.close()
cv2.destroyAllWindows()

print(
    f"✅ Done: {count} images saved to '{label}' folder"
)