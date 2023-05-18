import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import hilbert, butter, sosfilt

from common import Signal, CosenoidSignal, Audio


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
            message = "sinal and carrier don't have the same sample rate."
            raise ValueError(message)
        return modulator_sample_rate

    def _get_length(self):
        if self.modulator.signal.length != self.carrier.signal.length:
            message = "sinal and carrier don't have the same length."
            raise ValueError(message)
        return self.modulator.signal.length


class DSBCoherentDemodulation:
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


class SSBDemodulation:
    demodulated: Signal

    def __init__(self, modulated_signal,
                 filter_order, filter_cutoff_frequency,
                 modulation_type='lsb'):
        self.demodulated = self._demodulated_signal_coherent(
            modulated_signal,
            filter_order, filter_cutoff_frequency,
            modulation_type
        )

    def _demodulated_signal_coherent(self, modulated_signal,
                                     order, cuttoff_frequency):
        multiplied = self._demodulation_multiplying_stage(modulated_signal)
        sos = butter(order, cuttoff_frequency,
                     output='sos', fs=self.sample_rate)
        filtered = sosfilt(sos, multiplied)
        filtered = filtered - np.mean(filtered)
        return Signal(filtered, self.sample_rate, self.length)

    def _demodulation_multiplying_stage(self, modulated_signal):
        sincronizing_signal = CosenoidSignal(
            self.carrier.frequency,
            self.sample_rate,
            self.length
        )
        sincronizing_cosine = sincronizing_signal.signal.data_array
        return None


class DSBModulatedSignal:
    k_a: float
    modulated: Signal
    demodulated: dict[str, Signal]

    def __init__(self, modulator, carrier, k_a,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(modulator, carrier)
        self.k_a = self._get_k_a(k_a)
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(filter_order,
                                                    filter_cutoff_frequency)

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
        coherent = DSBCoherentDemodulation(
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


class DSBSCModulation:
    modulated: Signal
    demodulated: dict[str, Signal]

    def __init__(self, signal, carrier,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(signal, carrier)
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(filter_order,
                                                    filter_cutoff_frequency)

    def _get_signal(self):
        modulator_data_array = self.modulator.signal.data_array
        carrier_data_array = self.carrier.signal.data_array
        data_array = modulator_data_array * carrier_data_array
        return Signal(data_array, self.sample_rate, self.length)

    def _demodulated_signal(self, order, cutoff_frequency):
        coherent = DSBCoherentDemodulation(self.modulated,
                                           order, cutoff_frequency)
        coherent_signal = coherent.demodulated
        return coherent_signal


class ModulatedSignal:
    modulated: Signal
    demodulated_coherent: Signal

    def __init__(self, signal, carrier, k_a,
                 filter_order, filter_cutoff_frequency):
        self.modulation = Modulation(signal, carrier)
        self.modulated = self._get_signal()
        self.demodulated = self._demodulated_signal(
            filter_order, filter_cutoff_frequency
            )

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
        modulator_hilbert_array = np.imag(hilbert(modulator_data_array))
        #data_array = 
        #return Signal(data_array, self.sample_rate, self.length)

    def _multiplying_stage(self):
        sincronizing_signal = CosenoidSignal(
            self.carrier.cosenoid_frequency,
            self.sample_rate,
            self.length
        )
        sincronizing_cosine = sincronizing_signal.signal.data_array
        return self.modulated.data_array * 2 * sincronizing_cosine

    def _demodulated_signal(self, order, cuttoff_frequency):
        multiplied = self._multiplying_stage()
        sos = butter(order, cuttoff_frequency,
                     output='sos', fs=self.sample_rate)
        filtered = sosfilt(sos, multiplied)
        filtered = filtered - np.mean(filtered)
        return Signal(filtered, self.sample_rate, self.length)