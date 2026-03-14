# 🫁 Pneumonia Detection from Chest X-Ray

An end-to-end Deep Learning project that utilizes a Custom Convolutional Neural Network (CNN) to detect and classify instances of Pneumonia from chest X-ray images. The project includes a model training script, a confusion matrix visualization script, and a responsive web application built with Streamlit.

## 🌟 Features
- **Deep Learning Model:** A custom-built, lightweight CNN model using TensorFlow & Keras.
- **Accurate Detection:** Classifies chest X-rays into **Normal** or **Pneumonia**.
- **Interactive Web App:** Easy-to-use Streamlit interface for seamless image uploads and real-time predictions.
- **Evaluation Metrics:** Scripts provided to evaluate model accuracy and generate visually appealing Confusion Matrices with Seaborn.

## 📂 Project Structure

```bash
📦 Project - Pneumonia Detection
├── app.py # The Streamlit web application frontend
├── train_model.py # Script used to build and train the custom CNN model
├── generate_confusion_matrix.py # Script to plot a confusion matrix from test data
├── requirements.txt # Python dependencies
├── chest_xray/ # Dataset directory (must contain 'train' and 'test' folders)
└── pneumonia_model_custom.keras # The saved trained model (generated after training)
```

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed (preferably version 3.8 to 3.11). 

### 2. Installation
Clone or navigate to the project directory and install the necessary dependencies using `pip`:
```bash
cd "d:\CODING\Project - Pneumonia Detection"
pip install -r requirements.txt
```

### 3. Dataset
Download the **Chest X-Ray Images (Pneumonia)** dataset (commonly found on Kaggle) and place it in a folder named `chest_xray/` at the root of the project directory. Ensure the structure looks like this:
```
chest_xray/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
```

### 4. Training the Model (Optional)
If you do not have the pre-trained `pneumonia_model_custom.keras`, or if you wish to train the model from scratch, run the training script:
```bash
python train_model.py
```
This will train the CNN using the dataset, evaluate it on the test set, and output the customized `.keras` model file.

### 5. Evaluating the Model
To generate a classification report and a beautiful Seaborn confusion matrix plot (`confusion_matrix_improved.png`), run:
```bash
python generate_confusion_matrix.py
```

### 6. Starting the Application
To launch the Streamlit graphical user interface, simply run:
```bash
streamlit run app.py
```

This will automatically open a local web server (typically at `http://localhost:8501`) inside your browser. You can upload an image, and the model will provide a prediction along with confidence levels.

## 🛠️ Technologies Used
- **TensorFlow & Keras:** For designing and training the deep learning model.
- **Streamlit:** For quickly building an interactive, responsive front-end.
- **Seaborn & Matplotlib:** For plotting model metrics and matrices.
- **NumPy & Pillow (PIL):** For image processing and array manipulation.

## 📝 License
This project is open-source and free to be studied, customized, and shared.
