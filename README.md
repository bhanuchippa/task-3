# 🧠 AI Image Classifier

> A deep learning image classification web application powered by **MobileNetV2 Transfer Learning** and **TensorFlow**.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Transfer Learning Explained](#-transfer-learning-explained)
- [Key Features](#-key-features)
- [Technologies Used](#️-technologies-used)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Model Performance](#-model-performance)
- [Screenshots](#️-screenshots)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Project Overview

The **AI Image Classifier** is an end-to-end deep learning application that classifies images into **10 categories** using the CIFAR-10 dataset. It leverages **transfer learning** with MobileNetV2 (pretrained on ImageNet) to achieve high accuracy with efficient training, and serves predictions through a modern Flask web interface.

**What it does:**

- Accepts user-uploaded images via a sleek drag-and-drop web interface.
- Classifies images into one of 10 CIFAR-10 categories: **Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck**.
- Displays the **top-3 predictions** with animated confidence bars.
- Provides full training metrics including accuracy/loss graphs and a confusion matrix.

Built as **Internship Task 3** — demonstrating proficiency in deep learning, transfer learning, model evaluation, and full-stack ML deployment.

---

## 🧪 Transfer Learning Explained

**Transfer learning** is a machine learning technique where a model trained on one task is reused as the starting point for a model on a different task. Instead of training a deep neural network from scratch (which requires massive datasets and compute), we take a model that has already learned rich visual features from millions of images and adapt it to our specific classification problem.

### Why MobileNetV2?

| Advantage | Description |
|-----------|-------------|
| **Lightweight** | Only 3.4M parameters — fast training and inference |
| **Accurate** | Strong performance on ImageNet (71.8% top-1 accuracy) |
| **Mobile-friendly** | Designed for edge devices and resource-constrained environments |
| **Efficient** | Inverted residual blocks with linear bottlenecks reduce computation |

### Two-Phase Training Strategy

The model is trained in two phases to maximize performance:

1. **Phase 1 — Feature Extraction**: Freeze the entire MobileNetV2 base and train only the custom classification head. This lets the new layers learn to map pretrained features to CIFAR-10 classes without disrupting the base model's learned representations.

2. **Phase 2 — Fine-Tuning**: Unfreeze the top layers of MobileNetV2 and retrain the entire model with a very low learning rate. This allows the base model to adapt its high-level features specifically to CIFAR-10.

```
┌─────────────────────────────────────────────────────┐
│                Training Pipeline                     │
│                                                      │
│  ImageNet Weights ──► Frozen MobileNetV2 Base        │
│                              │                       │
│                              ▼                       │
│                     Custom Classification Head       │
│                     (GlobalAvgPool + Dense + Dropout) │
│                              │                       │
│                              ▼                       │
│              Phase 1: Feature Extraction (Frozen)    │
│                              │                       │
│                              ▼                       │
│              Phase 2: Fine-Tuning (Top Layers)       │
│                              │                       │
│                              ▼                       │
│                    Final Trained Model                │
└─────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- **MobileNetV2 Transfer Learning** — Pretrained on ImageNet for superior feature extraction
- **CIFAR-10 Dataset** — 60,000 images across 10 classes (50K train / 10K test)
- **Data Augmentation & Regularization** — Random flips, rotations, zoom, and dropout to prevent overfitting
- **Real-Time Image Classification** — Upload any image and get instant predictions
- **Top-3 Predictions with Confidence Scores** — See the model's certainty for each class
- **Animated Confidence Bars** — Visual feedback with smooth progress bar animations
- **Drag & Drop Image Upload** — Modern upload zone with file preview
- **Training Metrics Visualization** — Accuracy and loss curves plotted after training
- **Confusion Matrix** — Detailed per-class performance breakdown
- **Responsive Dark-Themed UI** — Sleek interface that works on desktop and mobile
- **Cross-Platform Compatible** — Runs on Windows, macOS, and Linux

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.8+** | Backend programming language |
| **TensorFlow / Keras** | Deep learning framework for model building and training |
| **MobileNetV2** | Pretrained CNN model (transfer learning base) |
| **Flask** | Lightweight web framework for serving predictions |
| **NumPy** | Numerical computing and array operations |
| **Pandas** | Data manipulation and analysis |
| **OpenCV** | Image preprocessing and transformation |
| **Matplotlib** | Training metrics graph generation |
| **Seaborn** | Statistical visualization (confusion matrix heatmap) |
| **Scikit-learn** | Model evaluation metrics and classification reports |
| **Pillow** | Image file handling and format conversion |
| **HTML / CSS / JS** | Frontend interface with responsive design |

---

## 📁 Project Structure

```
AI_Image_Classifier/
│
├── app.py                      # Flask web application (main entry point)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation (this file)
├── QUICK_START.md              # Quick start guide
├── run/                        # Project runner and scripts
│   ├── run.py                  # Python setup & runner script
│   ├── run.bat                 # Windows batch launcher wrapper
│   └── run.sh                  # macOS/Linux bash launcher wrapper
│
├── model/
│   ├── train_model.py          # Model training script (MobileNetV2)
│   └── mobilenet_model.h5      # Trained model weights (generated)
│
├── static/
│   ├── css/
│   │   └── style.css           # Application styles (dark theme)
│   ├── js/
│   │   └── script.js           # Frontend logic (drag & drop, predictions)
│   └── uploads/                # Temporary uploaded images (generated)
│
├── templates/
│   └── index.html              # Main web page template
│
├── outputs/
│   ├── accuracy_plot.png       # Training accuracy graph (generated)
│   ├── loss_plot.png           # Training loss graph (generated)
│   └── confusion_matrix.png   # Confusion matrix heatmap (generated)
│
└── dataset/                    # CIFAR-10 data (auto-downloaded)
```

---

## 🚀 Installation & Setup

### Prerequisites

- **Python 3.8** or higher
- **pip** package manager
- **4GB+ RAM** recommended (8GB+ for faster training)
- GPU is optional — training works on CPU but takes longer

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AI_Image_Classifier.git
cd AI_Image_Classifier
```

### Step 2: Run the Automated Launcher (Recommended)

The project includes an automated script that creates a virtual environment, installs dependencies, handles training check/setup, and starts the web application in one command.

#### Windows
```bash
run\run.bat
```

#### macOS / Linux
```bash
bash run/run.sh
```

---

### Step 2 (Alternative): Manual Installation & Setup

If you prefer to set up the environment manually:

#### 1. Create and Activate Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Train the Model
```bash
python model/train_model.py
```

This will:
1. **Download** the CIFAR-10 dataset automatically (~170 MB).
2. **Train** the MobileNetV2 model through feature extraction and fine-tuning phases (15–25 epochs).
3. **Save** the trained model to `model/mobilenet_model.h5`.
4. **Generate** training visualizations in the `outputs/` directory.

#### 4. Run the Application
```bash
python app.py
```

Open your browser and navigate to:

```
http://localhost:5000
```

---

## 📊 Model Performance

- **Expected Test Accuracy**: ~90–93%
- **Architecture**: MobileNetV2 base + Global Average Pooling + Dense (256, ReLU) + Dropout + Dense (10, Softmax)
- **Regularization**: Data augmentation (flips, rotation, zoom), Dropout (0.5), Batch Normalization
- **Optimizer**: Adam with learning rate scheduling (lower LR during fine-tuning)
- **Loss Function**: Categorical Cross-Entropy

### Sample Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95% |
| Validation Accuracy | ~91–93% |
| Test Accuracy | ~90–93% |

**Accuracy Graph**: Shows training accuracy climbing steadily with validation accuracy tracking closely — a sign of good generalization with minimal overfitting.

**Loss Graph**: Displays training and validation loss both decreasing over epochs, with the gap between them remaining small due to effective regularization.

**Confusion Matrix**: Reveals per-class performance — categories like *Ship*, *Automobile*, and *Truck* typically achieve the highest accuracy, while visually similar classes like *Cat* vs. *Dog* or *Deer* vs. *Horse* may show some confusion.

---

## 🖼️ Screenshots

> 📸 **Note**: Add screenshots after running the application.

| Screenshot | Description |
|---|---|
| Upload Interface | Dark-themed drag-and-drop zone for image uploads |
| Prediction Results | Top-3 predictions displayed with animated confidence bars |
| Training Metrics | Accuracy/loss curves and confusion matrix heatmap |

---

## 🔮 Future Improvements

- [ ] Support more image categories beyond CIFAR-10
- [ ] Deploy to cloud platforms (AWS / GCP / Heroku)
- [ ] Add batch image processing for multiple uploads
- [ ] Implement model versioning and experiment tracking (MLflow)
- [ ] Add API documentation with Swagger / OpenAPI
- [ ] Progressive Web App (PWA) support for offline use
- [ ] Docker containerization for reproducible deployment
- [ ] Add Grad-CAM visualization to explain model predictions
- [ ] Support custom dataset training via the web interface
- [ ] Implement A/B testing between model architectures

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **[TensorFlow](https://www.tensorflow.org/)** — Open-source deep learning framework by Google
- **[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)** — Curated by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton
- **[MobileNetV2](https://arxiv.org/abs/1801.04381)** — Mark Sandler, Andrew Howard, et al. (Google Research)
- **[Flask](https://flask.palletsprojects.com/)** — Lightweight Python web framework by Pallets Projects
- **[Shields.io](https://shields.io/)** — Badge generation for open-source projects

---

<p align="center">
  Made with ❤️ for Internship Task 3
</p>
# task-3
