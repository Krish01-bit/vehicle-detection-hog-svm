Vehicle Detection Using HOG and SVM
This document explains how to set up and run the vehicle detection project.

✅ Requirements
Install the required Python libraries:
pip install numpy opencv-python scikit-image scikit-learn matplotlib
Python version recommended: Python 3.8+

📁 Dataset Setup
This project uses vehicle and non-vehicle image datasets such as:
GTI Vehicle Dataset
KITTI Dataset
Download the dataset and place it in folders like:
samples/
 ├── vehicles/
 └── non-vehicles/
Update the dataset paths inside examples.py or train.py if needed.

▶️ How to Run
Run the main file:
python examples.py
This will:
Load training images
Extract HOG, color, and spatial features
Train SVM classifier
Run vehicle detection on the test video
Detection results will be displayed with bounding boxes.

🎥 Test Video
A sample test video (test_video.mp4) is included in the repository.
You may replace it with your own road video if needed.

⚠️ Notes
Detection accuracy depends on training data quality.
Some false detections may occur due to background similarity.
This project uses classical machine learning, not deep learning.

🎯 Purpose
This project is intended for academic learning of:
Feature extraction techniques
Support Vector Machines
Sliding window object detection