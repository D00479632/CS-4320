# Machine Learning Projects Collection

This repository contains a collection of machine learning projects for different applications, including regression, classification, and image processing tasks.

## Project Structure

- **neural_network_classification/** - Neural network models for classification tasks
- **neural_network_regression/** - Neural network models for regression problems
- **neural_network_image_classification/** - CNN models for image classification
- **linear_regression/** - Linear regression implementations
- **classification_cross_validation/** - Cross-validation techniques for classification
- **linear_cross_validation/** - Cross-validation techniques for regression

## Requirements

The projects require the following dependencies:

```
pandas
tensorflow
tensorflow-macos
tensorflow-metal
keras
numpy
joblib
matplotlib
scikit-learn
```

You can install all dependencies by running:

```bash
pip3 install -r requirements.txt
```

## Project Details

### Neural Network Classification

A neural network implementation for classification problems using TensorFlow/Keras.

**Key files:**
- `preprocess.py` - Data preprocessing utilities
- `neural_net_train.py` - Training implementation
- `neural_net_predict.py` - Model prediction implementation
- `pipeline_elements.py` - Pipeline components

### Neural Network Regression

Implementation of neural networks for regression tasks.

### Neural Network Image Classification

CNN-based implementation for image classification, primarily using the CIFAR-10 dataset.

**Key components:**
- Model creation and evaluation
- Training pipeline with hyperparameter tuning
- Visualization of training metrics
- GPU utilization for training acceleration

### Linear Regression

Implementation of linear regression models with different optimization techniques.

### Cross-Validation Projects

The repository includes implementations of cross-validation techniques for both classification and regression problems:
- `classification_cross_validation/`
- `linear_cross_validation/`

## Getting Started

1. Clone this repository
2. Install the dependencies with `pip install -r requirements.txt`
3. Navigate to the specific project directory you're interested in
4. Follow the instructions in the project-specific README or process.md file sometimes there is a .bash file where you can see the process I did and how everything went.

## Documentation

Each project folder contains detailed documentation:
- PDF files with theory and implementation details
- `process.md` files outlining the development process as well as reports
- Example implementations with commented code
