from utils.audio import Audio

import numpy as np
import matplotlib.pyplot as plt


class Offset_Detect(Audio):
    def __init__(self, y, sr, n_fft, hop_len):
        super().__init__(y, sr, n_fft, hop_len)
        self.avg_db = self.avg_db_timeline(self.Sdb)


    # Offset Detection
    def offset_detect(self, on_sets, threshold=-30, split_part=10):
        spectral_flux = self.n_spectral_flux(split_part=split_part)
        off_sets = []
        for on_set in on_sets:
            # Find Negative Flux
            for idx in range(on_set, len(spectral_flux)):
                if (spectral_flux[idx] < 0):
                    neg_flux_start = idx
                    break

            # Find Offset
            for idx in range(neg_flux_start, len(spectral_flux)):
                if (spectral_flux[idx] == 0 and self.avg_db[idx] < threshold):
                    off_sets.append(idx)
                    break

        return off_sets


    # Split Sdb into multiple part and apply spectral flux on it
    def n_spectral_flux(self, threshold=.3, split_part=10):
        spectral_flux = np.zeros(self.Sdb.shape[1] - 1)
        for Sdb in self.split_sdb(split_part):
            clean_Sdb = self.remove_noise(Sdb) # Wipe out background noise
            spectral_flux += self.spectral_flux(clean_Sdb) # Sum up the spectral flux
        spectral_flux /= split_part # Get mean of the spectral flux
        # If the flux is less then threshold, then seen it as no difference.
        spectral_flux[spectral_flux > -threshold] = 0

        return spectral_flux


    # Sum the positive change in each frequency bin
    def spectral_flux(self, Sdb):
        return np.mean(self.half_wave_rectifier(np.diff(Sdb, axis=1)), axis=0)


    # Function that input a numpy and only remain negative value
    def half_wave_rectifier(self, x):
        x[x > 0] = 0
        return x


    # Mean dB of all frequecy range vs Time
    def avg_db_timeline(self, Sdb):
        return [np.mean(Sdb[:, idx]) for idx in range(Sdb.shape[1])]


    # Draw spectral flux plot
    def draw_spectral_flux(self, offsets=None, split_part=10):
        spectral_flux = self.n_spectral_flux(split_part=split_part)
        plt.plot(spectral_flux)
        plt.ylim(top=np.max(spectral_flux) * 1.3)
        if (offsets != None):
            plt.plot(offsets, spectral_flux[offsets], 'rx')

        plt.show()