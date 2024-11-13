# AlexNet Recreation in PyTorch

This project is a recreation of the AlexNet convolutional neural network architecture using PyTorch. AlexNet was a groundbreaking model that significantly advanced the field of computer vision and deep learning. This implementation serves as a learning exercise and demonstration of the key components and techniques used in AlexNet.

## Project Overview

AlexNet, introduced by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton in their 2012 paper "ImageNet Classification with Deep Convolutional Neural Networks," was a milestone in the development of deep learning for computer vision tasks. It achieved state-of-the-art performance on the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and paved the way for the widespread adoption of deep learning in various domains.

The main features of AlexNet include:
- Utilization of convolutional layers for feature extraction
- Incorporation of ReLU activation function for non-linearity
- Use of max pooling for spatial downsampling
- Application of dropout regularization to prevent overfitting
- Training on a large-scale dataset (ImageNet) with data augmentation techniques

## Implementation Details

This project implements the AlexNet architecture using PyTorch, a popular deep learning framework. The implementation follows the original paper closely, with minor modifications to adapt to the PyTorch framework and modern practices.

The key components of the implementation include:
- Definition of the AlexNet model architecture using PyTorch's `nn.Module` class
- Loading and preprocessing of the ImageNet dataset (or a subset) for training and evaluation
- Training loop with optimization using stochastic gradient descent (SGD)
- Evaluation of the trained model on a validation set
- Visualization of the learned features and activations

## Requirements

To run this project, you need the following dependencies:
- Python
- PyTorch
- torchvision
- NumPy
- Matplotlib (for visualization)

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/AndrewWladis/alexnet-recreation.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Download the ImageNet dataset or a subset of it
   - Preprocess the dataset and organize it into appropriate directories

4. Train the AlexNet model:
   ```
   python train.py
   ```

## Results

The trained AlexNet model achieves competitive performance on the ImageNet dataset. The evaluation script provides metrics such as accuracy, precision, recall, and F1 score. The visualization script allows for qualitative analysis of the learned features and activations.

## Acknowledgments

This project is inspired by the original AlexNet paper:
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.


