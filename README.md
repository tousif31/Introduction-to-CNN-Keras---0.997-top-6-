# 🧠 Introduction to CNN using Keras — 0.997 Accuracy (Top 6%)

## 📘 Overview
This project demonstrates how to build and train a **Convolutional Neural Network (CNN)** using **Keras** (TensorFlow backend) to achieve high accuracy on the **MNIST digit recognition** dataset.  
It was developed as part of a 7th semester deep learning project and achieved **99.7% test accuracy**, placing it in the **top 6%** of submissions on the Kaggle *Digit Recognizer* competition.

---

## 🚀 Features
- Implementation of a CNN from scratch using **Keras**
- Trained on the **MNIST / Digit Recognizer** dataset
- Achieved **0.997 (99.7%)** accuracy on the test set
- Includes **data preprocessing**, **augmentation**, and **model evaluation**
- Provided as a Jupyter Notebook for easy experimentation

---

## 🗂️ Project Structure
Introduction-to-CNN-Keras---0.997-top-6-/
│
├── introduction-to-cnn-keras-0.997-top-6.ipynb # Main Jupyter notebook
├── digit-recognizer.zip # Dataset archive (if included)
├── README.md # Project documentation
└── requirements.txt # Dependencies (optional)

yaml
Copy code

---

## ⚙️ Requirements
Make sure you have the following Python libraries installed:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn
Or, create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
📊 Dataset
The model uses the Kaggle Digit Recognizer dataset, which is based on the classic MNIST handwritten digits (28×28 grayscale images).
If the dataset isn’t included, download it from Kaggle:
👉 https://www.kaggle.com/c/digit-recognizer

🧩 Model Architecture
A simplified summary of the CNN structure:

Conv2D → ReLU → MaxPooling2D

Conv2D → ReLU → MaxPooling2D

Flatten → Dense (128) → Dropout → Dense (10, softmax)

Optimizer: Adam
Loss Function: categorical_crossentropy
Metrics: accuracy

📈 Results
Metric	Value
Training Accuracy	~99.8%
Validation Accuracy	~99.7%
Kaggle Public Score	0.997 (Top 6%)

🧠 Learnings
How CNNs extract spatial features from image data

Role of dropout and normalization in preventing overfitting

How to tune hyperparameters (batch size, learning rate, etc.) for better performance

Importance of data augmentation for generalization

🔧 How to Run
🖥️ Run Locally
Clone the repository

bash
Copy code
git clone https://github.com/tousif31/Introduction-to-CNN-Keras---0.997-top-6-.git
cd Introduction-to-CNN-Keras---0.997-top-6-
Install dependencies

bash
Copy code
pip install -r requirements.txt
Launch Jupyter Notebook

bash
Copy code
jupyter notebook introduction-to-cnn-keras-0.997-top-6.ipynb
Run all cells to train and evaluate the model.
