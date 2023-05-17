import numpy as np
import matplotlib.pyplot as plt

from common import SenoidSignal, Audio

class ModulatedSignal:
    signal: SenoidSignal | Audio
    carrier: SenoidSignal
    k_a: float

    def __init__(self, signal, carrier, k_a):
        self.signal = signal
        self.carrier = carrier
        self.k_a = self._get_k_a(k_a)

    def _get_k_a(self, k_a):
        if k_a <= 0:
            raise ValueError('k_a must be positive.')
        return k_a

    @property
    def data_array(self):
        return (1 + self.k_a*self.signal.data_array) * self.carrier.data_array
    
    @property
    def time(self):
        return self.signal.time
    
    def plot_data(self):
        plt.figure()
        plt.plot(self.time, self.data_array)
        plt.xlabel('Tempo [s]')
        plt.ylabel('Amplitude')
        plt.show()