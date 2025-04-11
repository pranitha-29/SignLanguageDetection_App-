from tensorflow.keras.models import load_model # type: ignore

# Load the HDF5 model
model = load_model("sign_language_model.keras")

# Save in new Keras v3 format
model.save("sign_language_model.h5")

print("✅ Model successfully saved as sign_language_model.h5")
