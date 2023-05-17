import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

from common import Signal, SenoidSignal, Audio

class ModulatedSignal:
    sample_rate: float
    length: float
    modulated: Signal
    baseband: Signal
    modulator: SenoidSignal | Audio
    carrier: SenoidSignal
    k_a: float

    def __init__(self, signal, carrier, k_a):
        self.modulator = signal
        self.carrier = carrier
        self.sample_rate = self._get_sample_rate()
        self.length = self._get_length()
        self.k_a = self._get_k_a(k_a)
        self.modulated = self._get_signal()
        self.baseband = self._get_bandbase()

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

    def _get_bandbase(self):
        modulator_data_array = self.modulator.signal.data_array
        carrier_amplitude = self.carrier.amplitude
        data_array = carrier_amplitude * (1 + self.k_a*modulator_data_array)
        return Signal(data_array, self.sample_rate, self.length)