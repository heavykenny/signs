# Traffic Sign Recognition using Convolutional Neural Networks (CNN)

## Project Overview

This project develops a convolutional neural network (CNN) to classify traffic signs from the German Traffic Sign

Recognition Benchmark (GTSRB) dataset. The goal is to demonstrate the application of deep learning techniques in image recognition tasks, specifically for recognizing and classifying traffic signs from images.

## Features

- **Model Training**: Train a CNN model on the GTSRB dataset.
- **Model Evaluation**: Evaluate the model using accuracy, precision, and recall metrics.
- **Data Visualization**: Visualize training results with confusion matrices and accuracy/loss graphs.
- **Predictions**: Use the trained model to make predictions on new traffic sign images.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- Scikit-Learn

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/heavykenny/signs.git
   ```
2. Navigate to the cloned directory:
   ```bash
   cd signs
   ```

3. Install required Python packages:
     ```bash
     pip install -r requirements.txt
     ```

### Directory Structure

* datasets/: Contains the dataset files including train, test, and meta data downloaded from
  [Kaggle](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) to be extracted here.
* build/: Contains Trained models.
* static/: Contains static files for the web application.
    * images:
* templates/: Contains HTML templates for the web application.
    * index.htmls: Home page.
* app.py: Main script for running the web application.
* cnn-model.ipynb: Jupyter notebook for training the CNN model.

## Usage
To train the model, visualize the results, and make predictions, follow the steps below:
- Train the model using the Jupyter notebook `sign-cnn-model.ipynb`.
- Run the web application using the command:
    ```bash
    python app.py
    ```
- Open a web browser and navigate to test the model using the web application.

### Acknowledgments
German Traffic Sign Recognition Benchmark (GTSRB) for providing the dataset.
TensorFlow and Keras libraries for providing the tools necessary to build deep learning models.

#### Note: This is a school project for understanding the basics of CNN and how to implement it in a real-world scenario.
