# Face-Mask-Detection  

# 😷 Face Mask Detection using PyTorch

A deep learning project that detects whether a person is wearing a face mask using Convolutional Neural Networks and real-time webcam video processing via OpenCV.

Built with PyTorch and powered by transfer learning using MobileNetV2.

---

## 🚀 Project Overview

With the rise of health safety requirements, face mask detection has become an essential task in public spaces. This project detects whether people are wearing face masks in real time using a webcam or video feed.

### 🎯 Objectives:
- Classify face images as **"Mask"** or **"No Mask"**
- Perform real-time detection on webcam feed
- Build a full ML pipeline using PyTorch and OpenCV

---

## 🛠️ Features

- ✅ Real-time webcam detection using OpenCV
- ✅ Transfer learning with MobileNetV2
- ✅ Image preprocessing, training, and evaluation
- ✅ Binary classification: `With Mask` vs `Without Mask`
- ✅ Modular codebase for training and inference
- ✅ Supports CPU or GPU

---

## 📂 Project Structure

face-mask-detection/
├── dataset/
│ ├── with_mask/
│ └── without_mask/
├── model.py # Model architecture (MobileNetV2)
├── utils.py # Transforms and helper functions
├── train.py # Model training script
├── detect_mask.py # Real-time mask detection with webcam
├── download_images.py # (Optional) Downloads dataset from GitHub
├── README.md # Project documentation


---

## 🧪 Tech Stack

| Component     | Tech Used              |
|---------------|------------------------|
| Framework     | PyTorch                |
| Model         | MobileNetV2 (transfer learning) |
| CV & Inference| OpenCV, Haar Cascades  |
| Data Format   | Folder-based images    |
| Hardware      | CPU / GPU supported    |

---

## 🖥️ Installation

### 1. Clone this repo:

```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
If requirements.txt not available, install manually:

```bash
pip install torch torchvision opencv-python matplotlib tqdm scikit-learn
```
📥 Dataset
Organize your dataset like this:


dataset/
├── with_mask/
├── without_mask/
You can use the provided dataset_downloader.py script to automatically download example images from the public GitHub repo.

Or manually download from the original dataset repo.

🏋️‍♂️ Training the Model
```python
python train.py
```
This will:

Train a binary classifier on the dataset

Save the model as mask_detector.pth

🧠 Model Architecture
The model uses transfer learning:

Base Model: MobileNetV2 (pretrained on ImageNet)

Head:

GlobalAveragePooling

Dropout

Fully Connected Layer → Sigmoid (for binary classification)

You can customize this in model.py.

🎥 Real-Time Detection (Webcam)
Once the model is trained:

```python
python detect_mask.py
```
It will:

Detect faces using Haar Cascades

Predict mask status using your trained model

Draw a colored rectangle and label

Press Q to quit the window.

🔍 Sample Output

🌟 Further Improvements
 Add confidence scores to predictions

 Use MTCNN or Dlib for better face detection

 Train on larger and more diverse datasets

 Use CNN architectures like:

ResNet18

EfficientNet-B0

Vision Transformers (ViT)

 Deploy via Streamlit or Flask

 Add logging and metrics dashboard (e.g., WandB)

🧠 Other Algorithms for Face Mask Detection
Here are some alternative models and CV techniques you could explore:

Type	Examples
CNN Architectures	ResNet, EfficientNet, VGG, ViT
Object Detection	YOLOv5, SSD, Faster R-CNN
Face Detection	MTCNN, RetinaFace, Dlib
Classical CV	HOG + SVM, Haar Cascades (used here)

📜 Dataset Credits
The dataset used in this project is based on the open-source project:

🔗 GitHub: [dataset](https://github.com/chandrikadeb7/Face-Mask-Detection)

Thanks to the author for curating the dataset of masked and unmasked face images.

🤝 Contributing
Pull requests and suggestions are welcome!

📄 License
This project is open source and available under the MIT License.



Made with ❤️ by P.Vidya Praveen
