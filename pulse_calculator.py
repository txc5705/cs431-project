#!/usr/bin/env python3
# Author: Alvin Lin

import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

THRESHOLD = 10

class PulseCalculator:
    def __init__(self, t_start=0):
        self.t_start = t_start
        self.observations = None

    def add_observation(self, value, time):
        if self.observations is None:
            self.observations = np.array([value, time])
        else:
            self.observations = np.vstack((
                self.observations, [value, time - self.t_start]))

    def get_observations(self, window, sigma=5):
        if self.observations is None or self.observations.shape[0] < 10:
            return np.zeros(1), np.zeros(1)
        last_observation_t = self.observations[-1,1]
        selected = self.observations[:,1] > (last_observation_t - window)
        observations = self.observations[selected]
        values = gaussian_filter(observations[:,0], sigma=sigma)
        times = observations[:,1]
        return values, times

    def get_pulse(self, window=5000):
        values, times = self.get_observations(window)
        duration = (times[-1] - times[0]) / 1000
        if duration == 0:
            return 0
        peaks, _ = find_peaks(values)
        return len(peaks) / duration * 60

    def plot_pulse(self, window=math.inf):
        values, times = self.get_observations(window)
        peaks, _ = find_peaks(values, height=np.median(values))
        plt.figure(1)
        plt.plot(times, values)
        plt.plot(times[peaks], values[peaks], 'ro')
        plt.show()

if __name__ == '__main__':
    with open('face2.npy', 'rb') as f:
        face2 = np.load(f)
    calculator = PulseCalculator()
    calculator.observations = face2
    calculator.get_pulse()
