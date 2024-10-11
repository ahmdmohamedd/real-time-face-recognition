# Real-Time Face Detection and Recognition

## Overview
This project implements a real-time face detection and recognition system using OpenCV. The system detects faces from a live camera feed and compares them with a dataset of known faces, providing alerts for unrecognized individuals.

## Features
- Real-time face detection using Haar Cascades.
- Face recognition using the Local Binary Patterns Histograms (LBPH) algorithm.
- Alerts for unknown faces.
- User-friendly interface displaying names and warnings.

## Requirements
- Python 3.x
- OpenCV with contrib modules
- NumPy

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahmdmohamedd/real-time-face-recognition.git
   cd real-time-face-recognition
   ```

2. **Install required packages**:
   ```bash
   pip install opencv-python opencv-contrib-python numpy
   ```

## Usage
1. Prepare a dataset of known faces:
   - Create a directory named `dataset_faces`.
   - Inside this directory, create subdirectories for each person, with their name as the folder name.
   - Add images of the faces you want to recognize in the respective folders.

2. Run the face detection and recognition script:
   ```bash
   python face_detection_recognition.py
   ```

3. Press 'q' to exit the live feed.

## Acknowledgments
- OpenCV documentation for the face detection and recognition techniques.
- Contributions and support from the open-source community.
