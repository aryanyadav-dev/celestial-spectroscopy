import matplotlib.pyplot as plt

def plot_sample_spectra(spectra, labels, num_samples=5):
    """Plot sample spectra with labels"""
    plt.figure(figsize=(12, 8))

    for i in range(num_samples):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(spectra[i])
        plt.title(f"Sample {i+1}: {labels[i]}")
        plt.xlabel("Wavelength")
        plt.ylabel("Flux")

    plt.tight_layout()
    plt.show()
