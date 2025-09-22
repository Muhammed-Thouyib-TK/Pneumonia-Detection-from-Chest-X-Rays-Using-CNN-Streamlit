# 🩺 Pneumonia Detection from Chest X-Rays (CNN + Streamlit)

An end-to-end deep learning pipeline to classify chest X-rays as **Normal** or **Pneumonia**, deployed with **Streamlit** for interactive, real-time predictions.

---

## 🎯 Objective
Build a deep learning system that:
- Classifies chest X-rays as Normal or Pneumonia.
- Deploys a trained model with Streamlit for user-friendly predictions.

---

## 📊 Dataset & Preprocessing
- **Dataset**: Chest X-ray images with training, validation, and test splits.  
- **Preprocessing**:  
  - Convert to RGB  
  - Resize to **32×32**  
  - Normalize pixel values (/255.0)  
- **Data Augmentation**:  
  - Random zoom  
  - Shear  
  - Rotation  
  - Horizontal flip  

---

## 🧪 Model Architecture
- **CNN with 3 Conv–Pooling blocks**  
- **Dense layer** with ReLU activation + **Dropout (0.5)** for regularization  
- **Sigmoid layer** for binary classification (Normal vs Pneumonia)  
- Lightweight model with **~159K parameters** for efficient deployment  

---

## ⚙️ Deployment
- Model saved as **`Pneumonia_Prediction.h5`**  
- Streamlit app provides:  
  - Image upload  
  - Preprocessing pipeline  
  - Prediction with confidence score  
- Simple and intuitive UI  

---

## 🔍 Key Insights
- Compact CNNs can perform well in **medical imaging tasks**.  
- **Data augmentation** and **dropout** enhance generalization.  
- **Streamlit integration** ensures accessibility and interactivity.  
- **Future Scope**:  
  - Higher input resolution  
  - Transfer learning (ResNet / EfficientNet)  
  - Model explainability with Grad-CAM  

---

## 🛠 Tech Stack
- Python  
- TensorFlow / Keras  
- Streamlit  
- NumPy  
- Pillow  
- Matplotlib  

---

## 🚀 Live Demo
🔗 [Try the App](https://pneumonia-detection-web-apps.streamlit.app/)

---
