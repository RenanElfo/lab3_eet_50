import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal import butter, sosfilt


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
        return self.data_array.shape[0]

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
        plt.xlabel('Frequência [Hz]')
        plt.ylabel('Amplitude')
        plt.show()


class Audio:
    signal: Signal

    def __init__(self, audio_path, factor):
        self.signal = self._get_signal(audio_path, factor)

    def _get_signal(self, audio_path, factor):
        sample_rate, audio_stereo = wavfile.read(audio_path)
        data_array = audio_stereo[:, 0]
        sample_number = data_array.shape[0]
        length = sample_number / sample_rate
        time = np.linspace(0., length, sample_number)
        new_time = np.linspace(0., length, factor*sample_number)
        interpolated_data = np.interp(new_time, time, data_array)
        return Signal(interpolated_data, factor*sample_rate, length)


class CosenoidSignal:
    frequency: np.ndarray
    amplitude: float
    phase: float
    signal: Signal
    senoid: Signal

    def __init__(self, frequency, sample_rate, length,
                 *, amplitude=1, phase=0):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = np.deg2rad(phase)
        self.signal, self.senoid = self._get_signal(sample_rate, length)

    def _get_signal(self, sample_rate, length):
        sample_number = int(sample_rate * length)
        time = np.linspace(0., length, sample_number)
        cosine = np.cos(2*np.pi*self.frequency*time + self.phase)
        sine = np.sin(2*np.pi*self.frequency*time + self.phase)
        cosine_array = self.amplitude * cosine
        sine_array = self.amplitude * sine
        cosine_signal = Signal(cosine_array, sample_rate, length)
        sine_signal = Signal(sine_array, sample_rate, length)
        return cosine_signal, sine_signal


class Modulation:
    sample_rate: float
    length: float
    modulator: CosenoidSignal | Audio
    carrier: CosenoidSignal

    def __init__(self, signal, carrier):
        self.modulator = signal
        self.carrier = carrier
        self.sample_rate = self._get_sample_rate()
        self.length = self._get_length()

    def _get_sample_rate(self):
        modulator_sample_rate = self.modulator.signal.sample_rate
        carrier_sample_rate = self.carrier.signal.sample_rate
        if modulator_sample_rate != carrier_sample_rate:
            message = "modulator and carrier don't have the same sample rate."
            raise ValueError(message)
        return modulator_sample_rate

    def _get_length(self):
        if self.modulator.signal.length != self.carrier.signal.length:
            message = "modulator and carrier don't have the same length."
            raise ValueError(message)
        return self.modulator.signal.length


class CoherentDemodulation:
    demodulated: Signal

    def __init__(self, modulated_signal, carrier_frequency,
                 filter_order, filter_cutoff_frequency):
        self.demodulated = self._demodulated_signal_coherent(
            modulated_signal, carrier_frequency,
            filter_order, filter_cutoff_frequency
        )

    def _demodulated_signal_coherent(self, modulated_signal,
                                     carrier_frequency,
                                     order, cuttoff_frequency):
        multiplied = self._demodulation_multiplying_stage(
            modulated_signal, carrier_frequency
            )
        sample_rate = modulated_signal.sample_rate
        length = modulated_signal.length
        sos = butter(order, cuttoff_frequency,
                     output='sos', fs=sample_rate)
        filtered = sosfilt(sos, multiplied)
        filtered = filtered - np.mean(filtered)
        return Signal(filtered, sample_rate, length)

    def _demodulation_multiplying_stage(
            self, modulated_signal, carrier_frequency):
        sincronizing_signal = CosenoidSignal(
            carrier_frequency,
            modulated_signal.sample_rate,
            modulated_signal.length
        )
        sincronizing_cosine = sincronizing_signal.signal.data_array
        return modulated_signal.data_array * 2 * sincronizing_cosine
