import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from textblob import TextBlob  

class SpectralClassifier:
    def __init__(self, input_shape, num_classes, regression=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.regression = regression
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv1D(64, kernel_size=7, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(128, kernel_size=5, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax' if not self.regression else 'linear')
        ])
        
        if self.regression:
            model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model

    def preprocess_data(self, spectra, labels):
        """Preprocess spectral data and labels"""
        # Normalize spectra (Z-score normalization)
        spectra_normalized = (spectra - np.mean(spectra, axis=1, keepdims=True)) / \
                             (np.std(spectra, axis=1, keepdims=True) + 1e-8)

        # Reshape spectra for Conv1D input (adding a "channel" dimension)
        spectra_normalized = np.expand_dims(spectra_normalized, axis=-1)  # Adding the channel dimension
        
        # Encode labels (from string to integers, then to one-hot)
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        one_hot_labels = tf.keras.utils.to_categorical(encoded_labels)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            spectra_normalized, one_hot_labels, 
            test_size=0.2, random_state=42, stratify=one_hot_labels
        )

        return X_train, X_test, y_train, y_test, label_encoder

    def train_and_visualize(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train model and show progress"""
        print("\nTraining the model...")

        # EarlyStopping and ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('spectral_model_best.keras', monitor='val_accuracy', save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint, tensorboard]
        )

        # Visualize the training history
        self._plot_training_history(history)

    def _plot_training_history(self, history):
        """Plot training and validation accuracy/loss"""
        # Accuracy plot
        plt.figure(figsize=(14, 7))  # Increased figure size for better visibility
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Test Accuracy')
        plt.title('Model Accuracy', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Test Loss')
        plt.title('Model Loss', fontsize=14)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance on test set"""
        # Get predictions from the model
        y_pred = self.model.predict(X_test)
        
        # Get the class with the highest probability (argmax)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Confusion Matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        cr = classification_report(y_test_classes, y_pred_classes, target_names=['star', 'galaxy'])
        print("\nClassification Report:")
        print(cr)

        # Model Metrics: Precision, Recall, F1 Score (macro-average)
        precision = precision_score(y_test_classes, y_pred_classes, average='macro')
        recall = recall_score(y_test_classes, y_pred_classes, average='macro')
        f1 = f1_score(y_test_classes, y_pred_classes, average='macro')

        print("\nModel Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plot Confusion Matrix
        plt.figure(figsize=(8, 8))  # Larger figure for confusion matrix
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix", fontsize=14)
        plt.colorbar()
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, ['Star', 'Galaxy'], rotation=45, fontsize=12)
        plt.yticks(tick_marks, ['Star', 'Galaxy'], fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.xlabel('Predicted label', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Perform sentiment analysis after model evaluation
        self.sentiment_analysis(['star', 'galaxy'])

        return cm, cr, precision, recall, f1

    def sentiment_analysis(self, labels):
        """Perform sentiment analysis on given labels"""
        print("\nSentiment Analysis:")
        for label in labels:
            sentiment = "positive" if label == "star" else "neutral"
            print(f"Sentiment for '{label}': {sentiment}")

    def plot_sample_spectra(self, spectra, labels, num_samples=5):
        """Plot sample spectra with labels"""
        plt.figure(figsize=(12, 8))  

        for i in range(num_samples):
            plt.subplot(num_samples, 1, i + 1)  
            plt.plot(spectra[i])  
            plt.title(f"Sample {i + 1}: {labels[i]}", fontsize=14)  
            plt.xlabel("Wavelength", fontsize=12)  
            plt.ylabel("Flux", fontsize=12)  

        plt.tight_layout()  
        plt.show()  
