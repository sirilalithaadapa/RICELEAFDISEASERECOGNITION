# RICE LEAF DISEASE RECOGNITION with Convolutional Neural Networks

This project demonstrates image classification using two different Convolutional Neural Networks (CNNs) architectures. The models are trained and evaluated on an image dataset, with a focus on comparing the performance of a simple CNN versus a more complex CNN.

## Project Structure
- **data**: Directory containing training and validation datasets.
- **notebooks**: Jupyter notebooks for data preparation, model training, and evaluation.
- **models**: Saved models after training.

## Getting Started

### Prerequisites
- Google Colab or local environment with Jupyter Notebook
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Google Drive (for data storage and access)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sirilalithaadapa/RICELEAFDISEASERECOGNITION.git
    cd image-classification-cnn
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow numpy matplotlib
    ```

### Data Preparation

1. Store your training and validation datasets in Google Drive. Organize them into respective directories (e.g., `train` and `validation`).

### Training the Models

1. Open the `cnn_training.ipynb` notebook in Google Colab or Jupyter Notebook.
2. Mount Google Drive to access the datasets:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. Define the paths to your training and validation directories:
    ```python
    train_path = '/content/drive/My Drive/your_train_directory'
    validation_path = '/content/drive/My Drive/your_validation_directory'
    ```
4. Run the notebook cells to:
    - Initialize the data generators.
    - Define the CNN models.
    - Train the models.
    - Evaluate and compare the models.

### Model Architectures

#### Simple CNN
- **Convolutional Layers**: 3
- **Max-Pooling Layers**: 3
- **Dense Layers**: 2
- **Dropout**: None

#### Complex CNN
- **Convolutional Layers**: 3
- **Max-Pooling Layers**: 3
- **Dense Layers**: 2
- **Dropout**: Applied after each convolutional and dense layer

### Evaluation and Comparison

After training, the models' accuracy and loss on the training and validation datasets are plotted. A comparison plot highlights the differences in performance between the simple and complex models.

### Results

- The simple model might train faster and be less prone to overfitting on small datasets.
- The complex model, with additional dropout layers, is likely to generalize better on larger datasets.

### Conclusion

This project illustrates the importance of model complexity and regularization techniques in CNNs for image classification tasks.

### License

Distributed under the MIT License. See `LICENSE` for more information.

### Acknowledgements

- TensorFlow Documentation
- Keras Documentation
- Google Colab

### Contact

Project Link: [https://github.com/sirilalithaadapa/RICELEAFDISEASE](https://github.com/sirilalithaadapa/RICELEAFDISEASERECOGNITION)
