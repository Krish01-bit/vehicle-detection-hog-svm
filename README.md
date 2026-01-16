##🚗 Vehicle Detection Using HOG and SVM

##📌 Project Overview
```
This project implements a classical computer vision and machine learning approach for detecting vehicles in road videos using Histogram of Oriented Gradients (HOG) features and a Support Vector Machine (SVM) classifier.
The system is trained using labeled vehicle and non-vehicle images. During detection, a sliding window technique is applied on video frames, and detections are refined using heatmap thresholding to reduce false positives.
This project demonstrates feature-based object detection without using deep learning models.
```

##🧠 Methodology
```
The vehicle detection pipeline consists of the following stages:
Image Preprocessing
    Resize images
    Convert RGB to YCrCb color space
Feature Extraction
    HOG features (shape and edge information)
    Color histogram features
    Spatial features
Feature Scaling
    Normalize feature values for SVM training
SVM Training
    Binary classification: Vehicle vs Non-Vehicle
Sliding Window Detection
    Scan video frames at multiple positions and scales
Heatmap & Thresholding
    Combine multiple detections and suppress false positives
Bounding Box Output
    Draw rectangles around detected vehicles
```

##📁 Project Structure
```
svm-vehicle-detector/
│
├── descriptor.py        # Feature extraction functions (HOG, color, spatial)
├── train.py             # Dataset processing and SVM training
├── slidingwindow.py     # Sliding window generation
├── detector.py          # Heatmap, labeling, and bounding box drawing
├── examples.py          # Main execution script
├── simple.py            # Small test / utility script (optional)
│
├── images/              # Output and visualization images for report
│   ├── preprocessing.png
│   ├── hog_visualization.png
│   └── detection_output.png
│
├── test_video.mp4       # Sample test video
├── README.md
└── INSTRUCTIONS.md
```

##📊 Dataset
```
The project uses publicly available vehicle datasets:
GTI Vehicle Image Database
KITTI Vision Benchmark Suite
These datasets contain labeled vehicle and non-vehicle images used for training the SVM classifier.
```

##⚠️ Due to large size and licensing reasons, the dataset is not included in this repository.
You can download the dataset from:
https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013

##▶️ How to Run the Project
1. Install Required Libraries
pip install numpy opencv-python scikit-image scikit-learn matplotlib
2. Set Dataset Paths
Update the dataset folder paths in examples.py or train.py to point to:
Vehicle images folder
Non-vehicle images folder
3. Train and Run Detection
python examples.py
This will:
Extract features
Train SVM classifier
Run vehicle detection on the test video

##✅ Output
```
The output video frames display:
Green bounding boxes around detected vehicles
Heatmap-based filtering to reduce false positives
Sample output images are available in the images/ folder.
``` 
##⚠️ Limitations
```
Sensitive to lighting and background patterns
Detection accuracy depends on dataset quality
Slower than deep learning methods
May produce false positives
This approach is mainly for educational and academic purposes.
```
##🚀 Future Improvements
```
Replace SVM with deep learning detectors (YOLO, SSD, Faster R-CNN)
Train on larger datasets
Improve multi-scale detection
Optimize speed for real-time applications
```
##👨‍🎓 Academic Note
```
This project was implemented as part of a college mini-project to understand:
Feature extraction techniques
Classical machine learning classifiers
Object detection pipelines
```
