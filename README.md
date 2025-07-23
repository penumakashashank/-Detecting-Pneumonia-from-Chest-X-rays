# 🩺 Pneumonia Detection from Chest X-rays using CNN

## 📌 Project Overview

This project uses Convolutional Neural Networks (CNNs) to classify chest X-ray images as either **Pneumonia** or **Normal**. It is part of a medical image classification task aimed at aiding early diagnosis through deep learning.

---

## 🧠 Objective

- Build a CNN model to automatically detect pneumonia from chest X-ray images.
- Use **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix** for performance evaluation.

---

## 📁 Dataset

**Dataset Used:** [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Contains over 5,000 chest X-ray images categorized into:
  - `PNEUMONIA`
  - `NORMAL`

- Folder structure:
chest_xray/
train/
test/
val/


---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- Google Colab

---

## 🧪 Data Preprocessing

- Resize images to **150x150**
- Normalize pixel values to **[0,1]**
- Apply data augmentation using `ImageDataGenerator`:
- Rotation
- Zoom
- Shear
- Horizontal Flip

---

## 🧱 CNN Architecture

- 3 Convolution + MaxPooling layers
- Flatten
- Dense layer with Dropout
- Sigmoid output for binary classification

---

## 📈 Evaluation Metrics

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ Confusion Matrix

---

## 🖼️ Results & Visualization

- Plots of training vs validation accuracy and loss
- Confusion matrix heatmap
- Random test image predictions with labels

---

## ▶️ How to Run

1. Upload the Kaggle dataset ZIP or use Kaggle API to download.
2. Extract it to `/content/` in Google Colab.
3. Run the provided notebook/script to train and evaluate the model.

---

## 💡 Learnings

- Built a full pipeline for medical image classification using CNN
- Learned data augmentation and model evaluation
- Understood the importance of recall and F1 in medical diagnostics

---

## 📌 Note

This project is for **educational purposes only** and should not be used for real-world diagnosis without proper clinical validation.

---

## 📜 License

Dataset © Kaggle · Code © ['shashankpenumaka']
