import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert

from common import Signal, Modulation, CoherentDemodulation


class DSBModulatedSignal:
    k_a: float
    modulated: Signal
    demodulated: dict[str, Signal]

    def __init__(self, modulator, carrier, k_a,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(modulator, carrier)
        self.k_a = self._get_k_a(k_a)
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(
            filter_order, filter_cutoff_frequency
            )

    def _get_k_a(self, k_a):
        if k_a <= 0:
            raise ValueError('k_a must be positive.')
        return k_a

    def _get_signal(self):
        modulator_data_array = self.modulation.modulator.signal.data_array
        carrier_data_array = self.modulation.carrier.signal.data_array
        data_array = (1 + self.k_a*modulator_data_array) * carrier_data_array
        sample_rate = self.modulation.sample_rate
        length = self.modulation.length
        return Signal(data_array, sample_rate, length)

    def _demodulated_signal(self, order, cutoff_frequency):
        coherent = CoherentDemodulation(
            self.modulated, self.modulation.carrier.frequency,
            order, cutoff_frequency)
        coherent_signal = coherent.demodulated
        noncoherent_signal = self._demodulated_signal_noncoherent()
        return {'coherent': coherent_signal,
                'noncoherent': noncoherent_signal}

    def _demodulated_signal_noncoherent(self):
        abs_baseband = self._absolute_baseband()
        demodulated_array = abs_baseband - np.mean(abs_baseband)
        sample_rate = self.modulation.sample_rate
        length = self.modulation.length
        return Signal(demodulated_array, sample_rate, length)

    def _absolute_baseband(self):
        return np.abs(hilbert(self.modulated.data_array))


class DSBSCModulatedSignal:
    modulated: Signal
    demodulated: dict[str, Signal]

    def __init__(self, modulator, carrier,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(modulator, carrier)
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(
            filter_order, filter_cutoff_frequency
            )

    def _get_signal(self):
        modulator_data_array = self.modulation.modulator.signal.data_array
        carrier_data_array = self.modulation.carrier.signal.data_array
        data_array = modulator_data_array * carrier_data_array
        sample_rate = self.modulation.sample_rate
        length = self.modulation.length
        return Signal(data_array, sample_rate, length)

    def _demodulated_signal(self, order, cutoff_frequency):
        coherent = CoherentDemodulation(
            self.modulated, self.modulation.carrier.frequency,
            order, cutoff_frequency)
        coherent_signal = coherent.demodulated
        return coherent_signal


class SSBModulatedSignal:
    modulated: Signal
    demodulated_coherent: Signal

    def __init__(self, signal, carrier, modulation_type,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(signal, carrier)
        self.modulation_type = modulation_type
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(
            filter_order, filter_cutoff_frequency
            )

    def _is_usb(self):
        if self.modulation_type not in {'lsb', 'LSB', 'usb', 'USB'}:
            raise ValueError('Invalid modulation type.')
        return True if self.modulation_type in {'usb', 'USB'} else False

    def _get_signal(self):
        modulator_data_array = self.modulation.modulator.signal.data_array
        modulator_hilbert_array = np.imag(hilbert(modulator_data_array))
        cosine_data_array = self.modulation.carrier.signal.data_array
        sine_data_array = self.modulation.carrier.senoid.data_array
        main_data_array = modulator_data_array * cosine_data_array
        hilbert_data_array = modulator_hilbert_array * sine_data_array
        usb_invert_sign = (-1) ** int(self._is_usb())
        data_array = main_data_array + usb_invert_sign*hilbert_data_array
        sample_rate = self.modulation.sample_rate
        length = self.modulation.length
        return Signal(data_array, sample_rate, length)

    def _demodulated_signal(self, order, cutoff_frequency):
        coherent = CoherentDemodulation(
            self.modulated, self.modulation.carrier.frequency,
            order, cutoff_frequency)
        coherent_signal = coherent.demodulated
        return coherent_signal