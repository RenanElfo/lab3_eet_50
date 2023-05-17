import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert, butter, sosfilt

from common import Signal, CosenoidSignal, Audio

class ModulatedSignal:
    sample_rate: float
    length: float
    modulated: Signal
    demodulated_coherent: Signal
    demodulated_noncoherent: Signal
    modulator: CosenoidSignal | Audio
    carrier: CosenoidSignal
    k_a: float

    def __init__(self, signal, carrier, k_a,
                 filter_order, filter_cutoff_frequency):
        self.modulator = signal
        self.carrier = carrier
        self.sample_rate = self._get_sample_rate()
        self.length = self._get_length()
        self.k_a = self._get_k_a(k_a)
        self.modulated = self._get_signal()
        self.demodulated_coherent = self._demodulated_signal_coherent(
            filter_order, filter_cutoff_frequency
            )
        self.demodulated_noncoherent = self._demodulated_signal_noncoherent()

    def _get_k_a(self, k_a):
        if k_a <= 0:
            raise ValueError('k_a must be positive.')
        return k_a

    def _get_sample_rate(self):
        modulator_sample_rate = self.modulator.signal.sample_rate
        carrier_sample_rate = self.carrier.signal.sample_rate
        if modulator_sample_rate != carrier_sample_rate:
            message = "sinal and carrier don't have the same sample rate."
            raise ValueError(message)
        return modulator_sample_rate

    def _get_length(self):
        if self.modulator.signal.length != self.carrier.signal.length:
            message = "sinal and carrier don't have the same length."
            raise ValueError(message)
        return self.modulator.signal.length

    def _get_signal(self):
        modulator_data_array = self.modulator.signal.data_array
        carrier_data_array = self.carrier.signal.data_array
        data_array = (1 + self.k_a*modulator_data_array) * carrier_data_array
        return Signal(data_array, self.sample_rate, self.length)

    def _multiplying_stage(self):
        sincronizing_signal = CosenoidSignal(
            self.carrier.cosenoid_frequency,
            self.sample_rate,
            self.length
        )
        sincronizing_cosine = sincronizing_signal.signal.data_array
        return self.modulated.data_array * 2 * sincronizing_cosine

    def _demodulated_signal_coherent(self, order, cuttoff_frequency):
        multiplied = self._multiplying_stage()
        sos = butter(order, cuttoff_frequency,
                     output='sos', fs=self.sample_rate)
        filtered = sosfilt(sos, multiplied)
        filtered = filtered - np.mean(filtered)
        return Signal(filtered, self.sample_rate, self.length)

    def _demodulated_signal_noncoherent(self):
        abs_baseband = self._absolute_baseband()
        demodulated_array = abs_baseband - np.mean(abs_baseband)
        return Signal(demodulated_array, self.sample_rate, self.length)

    def _absolute_baseband(self):
        return np.abs(hilbert(self.modulated.data_array))