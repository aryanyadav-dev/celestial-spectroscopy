
# Spectral Classifier

This project is designed to classify spectral data using a convolutional neural network (CNN) implemented in TensorFlow/Keras. The model takes spectral data (typically flux vs. wavelength) as input and classifies it into predefined categories (e.g., stars and galaxies). The classifier works with 1D spectral data and leverages deep learning techniques to perform the classification.

## Features

- **Spectral Data Preprocessing**: The code preprocesses spectral data by normalizing it and reshaping it for the CNN input.
- **Convolutional Neural Network**: The classifier is built using 1D convolutional layers (Conv1D), pooling layers, dense layers, and dropout layers to prevent overfitting.
- **Model Training and Visualization**: The training process includes callbacks like EarlyStopping, ModelCheckpoint, and TensorBoard, and the training history (accuracy and loss) is visualized.
- **Evaluation**: The model evaluates the performance on test data using metrics such as precision, recall, F1 score, and provides a confusion matrix and classification report.
- **Sample Spectra Visualization**: The classifier also allows plotting sample spectra with their corresponding labels.

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- TensorFlow (preferably v2.x)
- NumPy
- Matplotlib
- Scikit-learn

You can install the necessary Python packages using `pip`:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

### 1. Prepare Your Data

You need spectral data and corresponding labels. The data should be in the form of a 2D array (`spectra`), where each row represents one sample and each column represents a wavelength point. The labels should be a list of category names (e.g., `['star', 'galaxy']`).

### 2. Create and Train the Model

```python
# Import necessary libraries
from spectral_classifier import SpectralClassifier

# Example spectral data and labels
spectra = ...  # Your spectral data as a 2D numpy array (samples x features)
labels = ...   # Your labels as a list of strings (e.g., ['star', 'galaxy'])

# Initialize the classifier
input_shape = (spectra.shape[1], 1)  # Shape of the input data (number of features, 1 channel)
num_classes = len(set(labels))  # Number of unique classes in the dataset
classifier = SpectralClassifier(input_shape=input_shape, num_classes=num_classes)

# Preprocess data
X_train, X_test, y_train, y_test, label_encoder = classifier.preprocess_data(spectra, labels)

# Train the model and visualize the results
classifier.train_and_visualize(X_train, y_train, X_test, y_test, epochs=20)
```

### 3. Evaluate the Model

Once the model is trained, you can evaluate its performance on the test data:

```python
# Evaluate the model
cm, cr, precision, recall, f1 = classifier.evaluate_model(X_test, y_test)
```

This will output a confusion matrix, classification report, and key evaluation metrics (precision, recall, F1 score).

### 4. Visualize Sample Spectra

You can also visualize some sample spectra with their corresponding labels:

```python
# Plot sample spectra
classifier.plot_sample_spectra(spectra, labels, num_samples=5)
```

## Methods

### `SpectralClassifier` Class

- **`__init__(input_shape, num_classes, regression=False)`**: Initializes the classifier with the input shape and number of classes.
- **`_build_model()`**: Builds the CNN model with Conv1D layers and appropriate activation and dropout layers.
- **`preprocess_data(spectra, labels)`**: Normalizes the spectra, reshapes it for Conv1D input, encodes labels, and splits the data into training and testing sets.
- **`train_and_visualize(X_train, y_train, X_val, y_val, epochs=20)`**: Trains the model and visualizes the training and validation accuracy and loss.
- **`_plot_training_history(history)`**: Plots the training and validation accuracy/loss.
- **`evaluate_model(X_test, y_test)`**: Evaluates the model on the test set and prints the confusion matrix, classification report, and other evaluation metrics (precision, recall, F1).
- **`plot_sample_spectra(spectra, labels, num_samples=5)`**: Visualizes sample spectra with their corresponding labels.

## How to Run the File

1. **Prepare the Data**: Ensure you have your spectral data and corresponding labels in the correct format.
   - Spectral data should be a 2D numpy array where each row corresponds to a sample, and each column represents the spectral features (e.g., wavelength points).
   - Labels should be a list of strings corresponding to the category of each sample (e.g., `['star', 'galaxy']`).

2. **Run the Classifier**:
   - Clone or download the project files.
   - In your Python script, import the `SpectralClassifier` class.
   - Load your spectral data and labels, then preprocess them using the `preprocess_data()` method.
   - Train the model using the `train_and_visualize()` method.
   - Evaluate the model using `evaluate_model()`.
   - Optionally, visualize sample spectra using `plot_sample_spectra()`.

3. **Run the Script**:
   - Save your Python script, ensuring it is in the same directory as `spectral_classifier.py` or adjust the import accordingly.
   - Run the script from the terminal or your Python IDE:
   ```bash
   python your_script.py
   ```

The script will train the model, visualize the progress, and evaluate its performance.

## Example Output

During training, you will see the following outputs:

- **Training and Validation Accuracy/Loss**: Plotted graphs showing how the model's accuracy and loss improve over epochs.
- **Confusion Matrix**: A heatmap of the confusion matrix showing the classification performance.
- **Classification Report**: Detailed performance metrics (precision, recall, F1 score) for each class.

### Example of Classification Report:
```
              precision    recall  f1-score   support

         Star       0.95      0.92      0.93       100
       Galaxy       0.89      0.93      0.91       100

    accuracy                           0.92       200
   macro avg       0.92      0.92      0.92       200
weighted avg       0.92      0.92      0.92       200
```

### Example of Confusion Matrix:
```
Confusion Matrix:
[[92  8]
 [ 7 93]]
```

## Notes

- Ensure that your spectral data is properly formatted and normalized.
- The `train_and_visualize()` method includes early stopping and model checkpointing to avoid overfitting and save the best model during training.
- The model is designed for classification tasks, but you can modify it for regression by setting the `regression=True` flag during initialization.
