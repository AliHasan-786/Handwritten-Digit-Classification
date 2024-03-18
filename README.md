# Handwritten-Digit-Classification
# Overview
This project focuses on the implementation of logistic regression for handwritten digit classification using the PyTorch framework. Logistic regression, despite its simplicity, serves as a powerful baseline for classification tasks. We utilize the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits. The goal is to correctly identify digits from 0 to 9 based on these images.

# Dependencies
Python
PyTorch 1.x
torchvision
matplotlib
To install the required libraries, run the following command:

pip install torch torchvision matplotlib

# Data
The MNIST dataset is used in this project, comprising 60,000 training images and 10,000 testing images of handwritten digits. Each image is a 28x28 pixel grayscale representation of a digit from 0 to 9.

# Preprocessing
Transformation: Images are transformed into tensors using transforms.ToTensor() for PyTorch compatibility.
Normalization: (Optional) Although not included in the provided code, it's recommended to normalize the dataset for better training efficiency.
Model Training & Evaluation
Model: A logistic regression model with a single linear layer is used. The model's input size is 784 (28x28), flattened from the original images, and the output size is 10, corresponding to the ten digit classes.
Optimizer: Stochastic Gradient Descent (SGD) with a learning rate of 0.001.
Loss Function: Cross-Entropy Loss, suitable for classification problems with multiple classes.
Training Loop: The model is trained over 50 epochs, with the loss and accuracy calculated for each epoch.

# Usage
To train and evaluate the model, run the script from your terminal. Ensure that all dependencies are installed and that the MNIST dataset is accessible to the script.

# Results
The model achieves an accuracy of approximately 86% after 50 epochs of training. The results can be improved by training for more epochs, adjusting the learning rate, or employing more complex models.

# Note
The accuracy achieved is a baseline; further optimizations and model complexities can enhance performance.
Experimenting with different learning rates, optimizers, and regularization techniques may yield better results.
Consider exploring more complex models like Convolutional Neural Networks (CNNs) for significant improvements in accuracy on image classification tasks like the MNIST dataset.



