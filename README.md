# Plant Disease Detection using Deep Learning

This project implements a deep learning model for detecting plant diseases from leaf images. The model can identify 38 different classes of healthy and diseased plant leaves with high accuracy.

## Dataset

The dataset used in this project is from [Kaggle's New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset). It contains:
- ~87,000 RGB images of healthy and diseased crop leaves
- 38 different classes of plant diseases
- 80/20 split ratio for training and validation
- Additional test set of 33 images for prediction purposes

## Project Structure

```
├── Train_plant_disease.ipynb   # Training notebook with model implementation
├── Test_Plant_Disease.ipynb    # Testing and evaluation notebook
├── main.py                     # Streamlit web application
├── requirement.txt             # Project dependencies
└── training_hist.json         # Training history and metrics
```

## Implementation Steps

1. **Data Preprocessing**
   - Image loading and preprocessing using TensorFlow
   - Image resizing to 128x128 pixels
   - Data augmentation for better model generalization

2. **Model Architecture**
   - Convolutional Neural Network (CNN) using TensorFlow
   - Multiple convolutional and pooling layers
   - Dense layers with dropout for regularization
   - Softmax output layer for 38-class classification

3. **Training**
   - Trained on 70,295 images
   - Validated on validation set
   - Used categorical crossentropy loss and Adam optimizer
   - Implemented early stopping to prevent overfitting


4.**Video Recording**
   The project includes a Streamlit-based web interface for real-time plant disease detection.


   https://github.com/user-attachments/assets/c95afd5d-9126-4509-9d19-52a352668bd1

## Requirements

```
tensorflow==2.10.0
scikit-learn==1.3.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
streamlit
librosa==0.10.1
```

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirement.txt
   ```

2. Run the Streamlit web application:
   ```bash
   streamlit run main.py
   ```

3. Upload a plant leaf image through the web interface to get disease predictions.

## Results

The model achieves high accuracy in detecting plant diseases, making it a valuable tool for early disease detection in agricultural applications. The web interface provides an easy-to-use platform for farmers and agricultural experts to quickly identify plant diseases.
