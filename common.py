import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile


class Signal:
    data_array: np.ndarray
    sample_rate: float
    length: float

    def __init__(self, data_array, sample_rate, length):
        self.data_array = data_array
        self.sample_rate = sample_rate
        self.length = length

    @property
    def sample_number(self):
        return int(self.sample_rate * self.length)

    @property
    def time(self):
        return np.linspace(0., self.length, self.sample_number)

    @property
    def fourier_frequencies(self):
        return rfftfreq(self.sample_number, 1/self.sample_rate)

    @property
    def fourier_array(self):
        return rfft(self.data_array, norm='forward')

    def plot_data(self, start_index=0, end_index=-1):
        time = self.time[start_index:end_index]
        data = self.data_array[start_index:end_index]
        plt.figure()
        plt.plot(time, data)
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


class CosenoidSignal:
    cosenoid_frequency: np.ndarray
    amplitude: float
    phase: float
    signal: Signal

    def __init__(self, cosenoid_frequency, sample_rate, length,
                 *, amplitude=1, phase=0):
        self.cosenoid_frequency = cosenoid_frequency
        self.amplitude = amplitude
        self.phase = np.deg2rad(phase)
        self.signal = self._get_signal(sample_rate, length)

    def _get_signal(self, sample_rate, length):
        sample_number = int(sample_rate * length)
        time = np.linspace(0., length, sample_number)
        cosine = np.cos(2*np.pi*self.cosenoid_frequency*time + self.phase)
        data_array = self.amplitude * cosine
        return Signal(data_array, sample_rate, length)


class Audio:
    signal: Signal

    def __init__(self, audio_path, scale):
        self.signal = self._get_signal(audio_path, scale)

    def _get_signal(self, audio_path, scale):
        sample_rate, audio_stereo = wavfile.read(audio_path)
        data_array = audio_stereo[:, 0]
        sample_number = data_array.shape[0]
        length = sample_number / sample_rate
        time = np.linspace(0., length, sample_number)
        new_time = np.linspace(0., length, scale*sample_number)
        interpolated_data = np.interp(new_time, time, data_array)
        return Signal(interpolated_data, scale*sample_rate, length)