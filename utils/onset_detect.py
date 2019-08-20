from utils.audio import Audio

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class Onset_Detect(Audio):
    def __init__(self, y, sr, n_fft, hop_len):
        super().__init__(y, sr, n_fft, hop_len)


    # Onset Detection
    def onset_detect(self, threshold=1, split_part=10):
        spectral_flux = self.n_spectral_flux(split_part=split_part)
        peaks, _ = find_peaks(spectral_flux, height=threshold)
        on_sets = []
        for peak in peaks:
            for idx in range(peak, 0, -1):
                if (spectral_flux[idx] == 0):
                    on_sets.append(idx)
                    break

        return on_sets


    # Split Sdb into multiple part and apply spectral flux on it
    def n_spectral_flux(self, threshold=.3, split_part=10):
        spectral_flux = np.zeros(self.Sdb.shape[1] - 1)
        for Sdb in self.split_sdb(split_part):
            clean_Sdb = self.remove_noise(Sdb) # Wipe out background noise
            spectral_flux += self.spectral_flux(clean_Sdb) # Sum up the spectral flux
        spectral_flux /= split_part # Get mean of the spectral flux
        # If the flux is less then threshold, then seen it as no difference.
        spectral_flux[spectral_flux < threshold] = 0

        return spectral_flux


    # Sum the positive change in each frequency bin
    def spectral_flux(self, Sdb):
        return np.mean(self.half_wave_rectifier(np.diff(Sdb, axis=1)), axis=0)


    # Function that input a numpy and only remain positive value
    def half_wave_rectifier(self, x):
        return (x + np.abs(x)) / 2


    # Draw spectral flux plot
    def draw_spectral_flux(self, onsets=None, split_part=10):
        spectral_flux = self.n_spectral_flux(split_part=split_part)
        plt.plot(spectral_flux)
        plt.ylim(top=np.max(spectral_flux) * 1.3)
        if(onsets != None):
            plt.plot(onsets, spectral_flux[onsets], 'rx')

        plt.show()