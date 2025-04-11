import cv2
import os

# Label for this session
label = "yes"  # change this each time
num_images = 200
save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0
x, y, w, h = 100, 100, 300, 300  # ROI coordinates

print(f"📸 Starting image collection for: {label}")

while count < num_images:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    roi = frame[y:y+h, x:x+w]

    # Display instructions
    cv2.putText(frame, f"Showing: {label} ({count+1}/{num_images})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Image Collection", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):  # Press 'c' to capture
        img_name = os.path.join(save_path, f"{label}_{count}.jpg")
        roi_resized = cv2.resize(roi, (64, 64))
        cv2.imwrite(img_name, roi_resized)
        print(f"✅ Captured image {count+1}")
        count += 1

    elif key == 27:  # ESC to quit
        break

print(f"✅ Finished collecting {count} images for '{label}'")
cap.release()
cv2.destroyAllWindows()
