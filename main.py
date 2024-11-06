from data_collector import SpectralDataCollector
from spectral_classifier import SpectralClassifier
from config import INPUT_SHAPE, NUM_CLASSES, NUM_SAMPLES, EPOCHS, MODEL_SAVE_PATH

def main():
    # Collect spectral data (SDSS or synthetic)
    collector = SpectralDataCollector()
    spectra, labels = collector.fetch_sdss_spectra(num_samples=NUM_SAMPLES)

    # Initialize the classifier with configuration from config.py
    classifier = SpectralClassifier(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = classifier.preprocess_data(spectra, labels)

    # Plot sample spectra (visualization of a few spectra)
    classifier.plot_sample_spectra(spectra, labels, num_samples=5)

    # Train the model and visualize the training process
    classifier.train_and_visualize(X_train, y_train, X_test, y_test, epochs=EPOCHS)

    # Evaluate the model and print metrics
    classifier.evaluate_model(X_test, y_test)

    # Save the trained model
    classifier.model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
