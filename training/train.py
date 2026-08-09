from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.h5"

IMG_SIZE = 64
EPOCHS = 30
BATCH_SIZE = 32

CLASS_LABELS = [
    "hello",
    "help",
    "i_love_you",
    "no",
    "please",
    "stop",
    "thanks",
    "yes"
]

CLASS_MAP = {label: i for i, label in enumerate(CLASS_LABELS)}

print("📁 Class order:", CLASS_MAP)


# ============================================================
# Load Dataset
# ============================================================

X = []
y = []

for label in CLASS_LABELS:
    label_path = DATA_DIR / label

    if not label_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {label_path}"
        )

    for image_path in label_path.iterdir():

        # Process image files only
        if not image_path.is_file():
            continue

        img = cv2.imread(str(image_path))

        if img is None:
            print(f"⚠️ Could not read image: {image_path}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(img)
        y.append(CLASS_MAP[label])


X = np.array(X, dtype=np.float32) / 255.0
y = tf.keras.utils.to_categorical(
    np.array(y),
    num_classes=len(CLASS_LABELS)
)

print(f"✅ Loaded {len(X)} images.")


# ============================================================
# Split Dataset
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"🧪 Training on {len(X_train)}, "
    f"testing on {len(X_test)}"
)


# ============================================================
# Build CNN Model
# ============================================================

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        len(CLASS_LABELS),
        activation="softmax"
    )
])


# ============================================================
# Compile Model
# ============================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ============================================================
# Callbacks
# ============================================================

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    save_format="h5",
    mode="max",
    verbose=1
)

earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Train Model
# ============================================================

history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[
        checkpoint_cb,
        earlystop_cb
    ]
)


# ============================================================
# Evaluate Model
# ============================================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(
    f"\n✅ Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# Classification Report & Confusion Matrix
# ============================================================

y_pred_probs = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)

y_true = np.argmax(
    y_test,
    axis=1
)

print("\n📊 Classification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=CLASS_LABELS
    )
)

print("\n📉 Confusion Matrix:")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)