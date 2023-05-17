import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from scipy.io import wavfile


class SenoidSignal:
    senoid_frequency: np.ndarray
    sample_rate: int
    signal_length: float
    amplitude: float
    phase: float

    def __init__(self, senoid_frequency, sample_rate, signal_length, *, amplitude=0, phase=0):
        self.senoid_frequency = senoid_frequency
        self.sample_rate = sample_rate
        self.signal_length = signal_length
        self.amplitude = amplitude
        self.phase = np.deg2rad(phase)

    @property
    def sample_number(self):
        return int(self.signal_length * self.sample_rate)

    @property
    def time(self):
        return np.linspace(0., self.signal_length, self.sample_number)

    @property
    def data_array(self):
        sine = np.sin(2*np.pi*self.senoid_frequency*self.time + self.phase)
        return self.amplitude * sine

    @property
    def fourier_frequencies(self):
        return fftshift(fftfreq(self.sample_number, 1/self.sample_rate))

    @property
    def fourier_array(self):
        return fftshift(fft(self.data_array, norm='forward'))

    def plot_data(self):
        plt.figure()
        plt.plot(self.time, self.data_array)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_fourier(self, start_index=0, end_index=-1):
        frequencies = self.fourier_frequencies[start_index:end_index]
        amplitude = np.abs(self.fourier_array)[start_index:end_index]
        plt.figure()
        plt.plot(frequencies, amplitude)
        plt.title('Absolute Value of Amplitude')
        plt.xlabel('Frequência [Hz]')
        plt.ylabel('Amplitude')
        plt.show()


class Audio:
    data_array: np.ndarray
    sample_rate: int

    def __init__(self, audio_path):
        self.sample_rate, audio_stereo = wavfile.read(audio_path)
        self.data_array = audio_stereo[:, 0]

    @property
    def sample_number(self):
        return self.data_array.shape[0]

    @property
    def audio_length(self):
        return self.sample_number / self.sample_rate

    @property
    def time(self):
        return np.linspace(0., self.audio_length, self.sample_number)

    @property
    def fourier_frequencies(self):
        return fftshift(fftfreq(self.sample_number, 1/self.sample_rate))

    @property
    def fourier_array(self):
        return fftshift(fft(self.data_array, norm='forward'))

    def plot_data(self):
        plt.figure()
        plt.plot(self.time, self.data_array)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.show()

    def plot_fourier(self, start_index=0, end_index=-1):
        frequencies = self.fourier_frequencies[start_index:end_index]
        amplitude = np.abs(self.fourier_array)[start_index:end_index]
        plt.figure()
        plt.plot(frequencies, amplitude)
        plt.title('Absolute Value of Amplitude')
        plt.xlabel('Frequência [Hz]')
        plt.ylabel('Amplitude')
        plt.show()