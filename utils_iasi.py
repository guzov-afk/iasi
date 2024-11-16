import numpy as np
from scipy.stats import skew,kurtosis
import scipy.signal
from scipy.signal import periodogram




def MAV(data):
    return np.mean(np.abs(data))

def ZCR(data):
	return ((data[:-1] * data[1:]) < 0).sum()


def WL(data):
	return  (abs(data[:-1]-data[1:])).sum()

def RMS(data):
    return np.sqrt(np.mean(data**2))

def SSC(data, alpha):
	return  ((data[:-1]-data[1:])*(data[:1]-data[1:]) > alpha).sum()

def Energy(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    spectral_energy = np.sum(spectral_amplitude)
    return skew(spectral_amplitude)

def spectralPower(data):
    spectrum = np.abs(np.fft.fft(data))**2
    total_power = np.sum(spectrum)
    return total_power

def HJ(data):
    fft_result = np.fft.fft(data)
    spectral_amplitude = np.abs(fft_result)
    dominant_frequencies_indices = np.argsort(spectral_amplitude)[::-1][:512]
    dominant_frequencies_values = np.fft.fftfreq(len(data))[dominant_frequencies_indices]
    spectral_energy = np.sum(spectral_amplitude)

    bandwidth = np.sum(spectral_amplitude[dominant_frequencies_indices]) / spectral_energy
    return bandwidth
def Skewness(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    return np.std(spectral_amplitude)



	



def emg_bandpass_filter(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, edges, 'bandpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def ampl_filter(data,threshold):
    for i in range(len(data)):
        if data[i] < threshold:
            data[i] = 0
    return data


def MNF(signal):
    fs = 512
    f, Pxx = periodogram(signal, fs=fs, window='hann')
    mnf = np.sum(f * Pxx) / np.sum(Pxx)

    return mnf



def iemg(data):
    spectral_amplitude = np.abs(np.fft.fft(data))
    spectral_energy = np.sum(spectral_amplitude)
    return kurtosis(spectral_amplitude)


