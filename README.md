# Sign Language Detection

A real-time sign language recognition application that uses computer vision and deep learning to recognize hand signs and convert them into spoken words.

## Features

- Real-time hand detection using MediaPipe
- One-hand sign recognition
- CNN-based image classification
- Real-time webcam detection using OpenCV
- 8 supported sign classes
- Confidence-based predictions
- Text-to-speech output using pyttsx3
- Desktop GUI application using Tkinter
- Separate scripts for model training and dataset collection

## Supported Signs

The current model recognizes the following signs:

- Hello
- Help
- I Love You
- No
- Please
- Stop
- Thanks
- Yes

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Tkinter
- Pillow
- pyttsx3

## Project Structure

```text
Sign Language Detection/
│
├── dataset/
│   ├── hello/
│   ├── help/
│   ├── i_love_you/
│   ├── no/
│   ├── please/
│   ├── stop/
│   ├── thanks/
│   └── yes/
│
├── models/
│   └── best_model.h5
│
├── results/
│
├── src/
│   └── app.py
│
├── training/
│   ├── train.py
│   └── collect_data.py
│
├── .gitignore
├── requirements.txt
└── README.md
## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/pranitha-29/SignLanguageDetection_App-.git
cd SignLanguageDetection_App-
Installation
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Virtual Environment

On Windows PowerShell:

.venv\Scripts\Activate.ps1
4. Install Dependencies
pip install -r requirements.txt
Run the Application

Make sure your webcam is connected and accessible.

From the project root directory, run:

python src/app.py

The application will open a desktop interface and use the webcam for real-time sign language recognition.

How It Works

The application follows the pipeline below:

Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Detection
   ↓
Hand Region Extraction
   ↓
64 × 64 Image Preprocessing
   ↓
CNN Model
   ↓
Sign Prediction
   ↓
Tkinter GUI
   ↓
Text-to-Speech
Recognition Process

The webcam captures live video frames.

OpenCV processes the video frames.

MediaPipe detects a single hand.

The hand region is extracted using the detected hand landmarks.

The extracted region is resized to 64 × 64 pixels.

The image is normalized before being passed to the model.

The trained CNN predicts the sign.

A confidence threshold is used to accept predictions.

The predicted sign is displayed in the desktop application.

The recognized sign is converted into speech using pyttsx3.

Model

The project uses a Convolutional Neural Network (CNN) trained to classify eight different hand signs.

The model architecture includes:

Convolutional layers
Batch Normalization
Max Pooling
Fully Connected Layer
Dropout
Softmax Output Layer

The trained model is stored at:

models/best_model.h5
Model Input
Image Size: 64 × 64
Channels: 3
Output Classes: 8
Supported Output Classes
hello
help
i_love_you
no
please
stop
thanks
yes
Dataset

The project currently contains 2,500 images distributed across eight sign classes.

The dataset is organized into separate folders according to the corresponding sign:

dataset/
├── hello/
├── help/
├── i_love_you/
├── no/
├── please/
├── stop/
├── thanks/
└── yes/

Each image is resized to 64 × 64 pixels during the training and prediction process.

Training

The model can be trained using the included training script:

python training/train.py

The training script performs the following steps:

Loads images from the dataset.
Maps each sign class to a numerical label.
Resizes images to 64 × 64 pixels.
Normalizes pixel values.
Splits the dataset into training and testing sets.
Builds the CNN architecture.
Trains the model using the Adam optimizer.
Uses ModelCheckpoint to save the best model.
Uses EarlyStopping to reduce unnecessary training.
Evaluates the trained model.
Generates a classification report.
Generates a confusion matrix.
Dataset Collection

Additional images can be collected using:

python training/collect_data.py

The collection script:

Uses the webcam to capture images.
Detects one hand using MediaPipe.
Extracts the hand region.
Resizes the extracted region to 64 × 64 pixels.
Saves the images into the corresponding dataset class folder.

The script is currently configured to collect images for the please class by default.

To collect images for another supported class, update the label value inside:

training/collect_data.py

For example:

label = "hello"
Application Controls

The desktop application provides:

Start button to begin recognition.
Stop button to stop recognition.
Live webcam display.
Hand landmark visualization.
Bounding box around the detected hand.
Predicted sign display.
Automatic voice output for recognized signs.
Requirements

The project is designed to run with Python 3.10 and uses the dependencies listed in:

requirements.txt

The main machine learning and computer vision libraries include:

TensorFlow
NumPy
OpenCV
MediaPipe
Scikit-learn
Pillow
pyttsx3
Future Improvements

Possible future improvements include:

Add more sign language classes.
Increase dataset size and diversity.
Improve recognition accuracy.
Improve performance under different lighting conditions.
Add multilingual speech output.
Improve the desktop user interface.
Add prediction history.
Add continuous sign-to-speech sentence generation.
Improve robustness against different backgrounds and hand positions.
Project Purpose

This project was developed as an educational and portfolio project to demonstrate the use of:

Computer Vision
Deep Learning
Image Classification
Hand Landmark Detection
Real-Time Video Processing
Speech Synthesis
Desktop Application Development
