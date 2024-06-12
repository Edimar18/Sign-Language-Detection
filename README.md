# Sign Language Detection using PyQt5 and OpenCV

This repository contains a PyQt5 application for detecting and recognizing American Sign Language (ASL) gestures using OpenCV and a pre-trained machine learning model using Random Forest Classifier.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

These instructions will help you set up and run the project on your local machine.

## Prerequisites

Before you begin, make sure you have the following installed:

- Python 3.x
- PyQt5
- OpenCV
- Mediapipe
- Numpy
- Pickle
- Pyttsx3
- TensorFlow (optional, if you want to train your own model)

## Installation

Clone the repository:

```bash
git clone https://github.com/Edimar18/Sign-Language-Detection.git
cd Sign-Language-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the application, execute the following command:

```bash
python main.py
```

The application will open a window with a video feed from a camera or a video file. You can use the buttons to control the functionality:

- "Detect Hand": Detects and displays hand landmarks in the video feed.
- "Show Cam": Displays the video feed from the camera.
- "Predict Hand": Predicts the sign language gesture based on the detected hand landmarks.
- "No Predict": Disables gesture prediction.
- "Add Data": Opens a dialog to enter a folder name for adding new training data.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.



