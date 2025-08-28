Image Classification with CIFAR-10

This project implements a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model classifies images into 10 different categories with high accuracy.

Project Overview

The CIFAR-10 image classification system uses a deep learning approach with convolutional neural networks to accurately classify images into 10 categories. The solution includes data preprocessing, model architecture design, training, evaluation, and visualization of results.

Features

Data Preprocessing: Normalizes pixel values and prepares the CIFAR-10 dataset

CNN Architecture: Implements a convolutional neural network with multiple layers

Model Training: Includes dropout regularization to prevent overfitting

Model Evaluation: Comprehensive evaluation using accuracy and loss metrics

Visualization: Generates training history plots and prediction visualizations

Model Saving: Saves the trained model for future use

Requirements

Python 3.7+

TensorFlow 2.x

matplotlib

numpy

Install dependencies with:

bash
pip install tensorflow matplotlib numpy
Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The 10 classes are:

airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck

Model Architecture

The CNN model consists of:

Conv2D layer with 32 filters (3x3)

MaxPooling2D layer (2x2)

Conv2D layer with 64 filters (3x3)

MaxPooling2D layer (2x2)

Conv2D layer with 64 filters (3x3)

Flatten layer

Dense layer with 64 units and ReLU activation

Dropout layer (0.5) for regularization

Output layer with 10 units (one for each class)

Usage
Run the script:

bash
python image_classification_cifar10.py
The script will:

Load and preprocess the CIFAR-10 dataset

Define and compile the CNN model

Train the model for 15 epochs

Evaluate model performance on test data

Generate visualizations of training history and predictions

Save the trained model for future use

Output Files
training_history.png: Training and validation accuracy/loss plots

predictions.png: Sample predictions with true and predicted labels

cifar10_cnn_model.h5: Trained model file

Performance
The model typically achieves:

Training accuracy: ~75-80%

Test accuracy: ~70-75%

These results can be improved by:

Training for more epochs

Using data augmentation

Implementing more complex architectures (e.g., ResNet, VGG)

Using transfer learning

Making New Predictions
After training, the model can be used to classify new images:

python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('cifar10_cnn_model.h5')

# Preprocess new image (resize to 32x32 and normalize)
def preprocess_image(image_path):
    img = Image.open(image_path).resize((32, 32))
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] != 3:  # Convert to RGB if needed
        img_array = np.stack((img_array,) * 3, axis=-1)
    return np.expand_dims(img_array, axis=0)

# Make prediction
new_image = preprocess_image('new_image.jpg')
predictions = model.predict(new_image)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Predicted class: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
Customization
You can modify the script to:

Adjust the number of epochs

Change the model architecture

Add data augmentation

Implement early stopping

Use different optimizers or learning rates

Apply transfer learning with pre-trained models

License
This project is open source and available under the MIT License.
