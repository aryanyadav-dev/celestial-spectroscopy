from flask import Flask, jsonify
from flask_cors import CORS
from data_collector import SpectralDataCollector
from spectral_classifier import SpectralClassifier
from config import INPUT_SHAPE, NUM_CLASSES, NUM_SAMPLES, EPOCHS, MODEL_SAVE_PATH
import logging
import numpy as np

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_metrics = {}
training_history = {}
sample_spectra = []

def run_spectral_classification():
    global model_metrics, training_history, sample_spectra
    
    try:
        collector = SpectralDataCollector()
        spectra, labels = collector.fetch_sdss_spectra(num_samples=NUM_SAMPLES)

        classifier = SpectralClassifier(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        
        X_train, X_test, y_train, y_test, label_encoder = classifier.preprocess_data(spectra, labels)

        sample_spectra = classifier.plot_sample_spectra(spectra, labels, num_samples=5)

        history = classifier.train_and_visualize(X_train, y_train, X_test, y_test, epochs=EPOCHS)
        if history is not None and hasattr(history, 'history'):
            training_history = history.history
        else:
            logger.warning("Training history is not available")
            training_history = {}

        model_metrics = classifier.evaluate_model(X_test, y_test)

        classifier.model.save(MODEL_SAVE_PATH)
        
        logger.info("Spectral classification completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during spectral classification: {str(e)}")
        raise

@app.route('/api/model-metrics')
def get_model_metrics():
    return jsonify(model_metrics)

@app.route('/api/training-history')
def get_training_history():
    return jsonify({
        'epochs': list(range(1, len(training_history.get('accuracy', [])) + 1)),
        'accuracy': training_history.get('accuracy', []),
        'loss': training_history.get('loss', []),
        'valAccuracy': training_history.get('val_accuracy', []),
        'valLoss': training_history.get('val_loss', [])
    })

@app.route('/api/spectral-data')
def get_spectral_data():
    return jsonify([
        {
            'wavelength': spectrum['wavelength'].tolist(),
            'flux': spectrum['flux'].tolist(),
            'label': label
        }
        for spectrum, label in sample_spectra
    ])

if __name__ == '__main__':
    run_spectral_classification()
    app.run(debug=True)