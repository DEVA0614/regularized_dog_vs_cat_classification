# Dog vs Cat Image Classification using CNN (GPU + Regularization)

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow and Keras** to classify images of **dogs and cats**. The model is trained with **data augmentation**, **L2 regularization**, and **dropout** to reduce overfitting and improve generalization. GPU support is enabled when available to speed up training.

---

## Features

* Binary image classification (Dog vs Cat)
* Automatic GPU detection and usage
* Removal of corrupted images from the dataset
* Data augmentation for better model performance
* L2 regularization and Dropout to prevent overfitting
* Training progress visualization (accuracy and loss)
* Single image prediction support
* Model saving for future use

---

## Dataset

* Dataset directory contains two folders:

  * `Dog`
  * `Cat`
* Images are resized to **150 × 150**
* Dataset split:

  * 80% Training
  * 20% Validation

Dataset path used:

```
/content/drive/MyDrive/PetImages
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* PIL (Python Imaging Library)
* Google Colab (recommended for GPU support)

---

## Model Architecture

* Convolutional layers with ReLU activation
* Max Pooling layers
* Flatten layer
* Fully connected Dense layer with L2 regularization
* Dropout layer (0.5)
* Output layer with Sigmoid activation

Loss Function:

* Binary Crossentropy

Optimizer:

* Adam

---

## Training Configuration

* Image size: 150 × 150
* Batch size: 128
* Epochs: 30
* Validation split: 20%

---

## How to Run the Project

1. Upload the dataset to Google Drive.
2. Update the dataset path in the code if required.
3. Run the script in Google Colab or a local environment with TensorFlow installed.
4. Training will start automatically after data preprocessing.
5. The trained model will be saved as:

```
dog_cat_model_regularized.h5
```

---

## Results

* Training and validation accuracy and loss are plotted after training.
* The model shows stable learning behavior due to regularization and data augmentation.

---

## Single Image Prediction

The project includes a function to predict a single image as either **Dog** or **Cat** after training. The image is resized, normalized, and passed to the trained model for prediction.

---

## Conclusion

This project demonstrates the practical implementation of a CNN-based image classification system using deep learning. It highlights the importance of data preprocessing, regularization techniques, and GPU acceleration in building efficient and accurate machine learning models.

