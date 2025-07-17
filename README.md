# CIFAR-100 CNN Classifier

CNN implementation for image classification on CIFAR-100 using PyTorch.

## Results

**Test Accuracy:** 61%  
**Training:** 20 epochs on Google Colab GPU  
**Dataset:** CIFAR-100 (100 classes, 32×32 RGB images)

## Implementation

Techniques used:

- **Batch normalisation** for training stability
- **Leaky ReLU** activation to prevent dying neurons  
- **Data augmentation** with random crops and horizontal flips
- **Adam optimiser** with 0.001 learning rate
- **Dropout regularisation** to reduce overfitting

Model accuracy improved from 11% to 61% over 20 epochs through iterative refinements.

## Architecture

```
Conv2d(3→64) + BatchNorm + LeakyReLU + MaxPool
Conv2d(64→128) + BatchNorm + LeakyReLU + MaxPool
Conv2d(128→256) + BatchNorm + LeakyReLU
Conv2d(256→256) + BatchNorm + LeakyReLU + MaxPool
Linear(4096→512) + LeakyReLU + Dropout(0.3)
Linear(512→100)
```

## Getting Started

```bash
jupyter notebook CompVision_A2.ipynb
```

**Requirements:** PyTorch, torchvision, CUDA-compatible GPU
