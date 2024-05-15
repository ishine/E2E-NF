import librosa
import numpy as np


def get_centroid(spectrum: np.ndarray, sr: int, n_fft: int) -> np.ndarray:
    return librosa.feature.spectral_centroid(S=spectrum, n_fft=n_fft, sr=sr)


# calculate spectral slope
def get_slope(spectrum: np.ndarray) -> np.ndarray:
    spectrum_mean = spectrum.mean(0, keepdims=True)
    centralized_spectrum = spectrum - spectrum_mean

    time_dimension = spectrum.shape[0]
    index = np.arange(0, time_dimension) - time_dimension / 2

    return np.dot(index, centralized_spectrum) / np.dot(index, index)


def get_energy(spectrum: np.ndarray) -> np.ndarray:
    return np.sum(spectrum**2, axis=0)
