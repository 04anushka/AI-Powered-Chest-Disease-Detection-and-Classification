# AI-Powered Chest Disease Detection and Classification

## Overview

This project focuses on automating the detection and classification of chest diseases from X-ray images using deep learning techniques. The model leverages a pre-trained ResNet50 architecture to classify X-ray images into four categories:
- **Class 0**: Healthy
- **Class 1**: COVID-19
- **Class 2**: Bacterial Pneumonia
- **Class 3**: Viral Pneumonia

The aim is to reduce the cost and time associated with manual diagnosis and enhance the accuracy of chest disease detection.

## Project Structure

- `data/`: Contains datasets for training and testing. The images are organized into subdirectories for each class.
- `notebooks/`: Jupyter notebooks for data exploration, visualization, and experimentation.
- `src/`: Source code for model training, evaluation, and prediction.
- `model/`: Saved model architecture and weights.
- `requirements.txt`: List of Python packages required for the project.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/chest-disease-detection.git
cd chest-disease-detection
pip install -r requirements.txt
```

## Data Preparation

1. **Download Dataset**: Place your X-ray images in the `data/Chest_X_Ray/` directory. Ensure that the images are organized into `train` and `test` subdirectories, with further subdirectories for each class.

2. **Directory Structure**:
   ```
   data/
     Chest_X_Ray/
       train/
         Healthy/
         Covid/
         Bacterial_Pneumonia/
         Viral_Pneumonia/
       test/
         Healthy/
         Covid/
         Bacterial_Pneumonia/
         Viral_Pneumonia/
   ```

## Training the Model

To train the model, use the provided script:

```bash
python src/train_model.py
```

This script will:
- Load and preprocess the data.
- Train a ResNet50-based model with custom classification layers.
- Save the model architecture and weights to the `model/` directory.

## Evaluating the Model

After training, evaluate the model on the test dataset:

```bash
python src/evaluate_model.py
```

This script will:
- Load the saved model.
- Evaluate its performance on the test set.
- Print accuracy, classification report, and confusion matrix.

## Usage

To make predictions on new X-ray images:

1. Place your images in the `data/new_images/` directory.

2. Run the prediction script:

```bash
python src/predict.py --image_dir=data/new_images/
```

This script will output the predicted class for each image.

## Results

- **Test Accuracy**: Approximately 77.5% to 80%
- **Confusion Matrix**: Shows class-wise performance and areas of confusion.

## Notes

- Ensure your data is properly labeled and organized as described.
- For further improvements, consider experimenting with additional data augmentation and hyperparameter tuning.

## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.

## Acknowledgments

- [ResNet50](https://arxiv.org/abs/1512.03385) for the pre-trained model.
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework.
