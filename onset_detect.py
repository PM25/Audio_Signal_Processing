import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import find_peaks
from scipy import signal
import math


class Onset_Detect:
    def __init__(self, y, sr, n_fft, hop_len):
        self.y = y
        self.SR = sr
        self.N_FFT = n_fft
        self.HOP_LEN = hop_len
        self.S = np.abs(librosa.core.stft(y, n_fft=self.N_FFT, hop_length=self.HOP_LEN))
        self.Sdb = librosa.amplitude_to_db(self.S, ref=1.0)
        self.freqs = librosa.core.fft_frequencies(n_fft=n_fft)


    # Plot the Spectrogram
    def spectrogram(self):
        plt.figure(figsize=(10, 4))
        plt.title("Mel Spectrogram")
        librosa.display.specshow(self.Sdb, y_axis='mel', x_axis='time', sr=self.SR, hop_length=self.HOP_LEN)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()


    # Normalization Function
    def normalize(self, x):
        x -= np.mean(x)
        x /= np.std(x)
        return x


    # Split Sdb into multiple part
    def split_sdb(self, split_part=3):
        part_size = math.ceil(len(self.freqs) / split_part)
        sdb_list = []
        for idx in range(part_size, len(self.freqs), part_size):
            sdb_list.append(self.Sdb[idx - part_size:idx])
        sdb_list.append(self.Sdb[idx:])

        return sdb_list


    # Onset Detection
    def onset_detect(self, split_part=3):
        spectral_flux = self.n_spectral_flux(split_part)
        peaks, _ = find_peaks(spectral_flux, height=.9)
        on_sets = []
        for peak in peaks:
            for idx in range(peak, 0, -1):
                if (spectral_flux[idx] == 0):
                    on_sets.append(idx)
                    break

        return on_sets


    # Smooth function
    def smooth(self, y, hann=15):
        win = signal.hann(hann)
        return signal.convolve(y, win, mode='same') / sum(win)


    # Draw vertical lines on the plot
    def draw_vlines(self, frame, x='time', color='w'):
        if(x == 'time'):
            y = librosa.frames_to_time(frame, sr=self.SR, hop_length=self.HOP_LEN)
        elif(x == 'frame'):
            y = frame

        plt.vlines(y, 0, self.freqs[-1], colors=color, linestyles='--')


    # Function that input a numpy and only remain positive value
    def half_wave_rectifier(self, x):
        return (x + np.abs(x)) / 2


    # Sum the positive change in each frequency bin
    def spectral_flux(self, Sdb):
        return np.mean(self.half_wave_rectifier(np.diff(Sdb, axis=1)), axis=0)


    # Split Sdb into multiple part and apply spectral flux on it
    def n_spectral_flux(self, split_part=3):
        spectral_flux = np.zeros(self.Sdb.shape[1] - 1)
        for Sdb in self.split_sdb(split_part):
            Sdb_norm = self.normalize(Sdb) # Noise will be in range of -1 ~ 1
            Sdb_norm -= 1 # Make all noise get below 0
            Sdb_norm[Sdb_norm <= 0] = 0 # Wipe out noise
            spectral_flux += self.spectral_flux(Sdb_norm)
        spectral_flux /= split_part
        spectral_flux = self.smooth(spectral_flux, 5)
        spectral_flux[spectral_flux < .1] = 0

        return spectral_flux


    # Draw spectral flux plot
    def draw_spectral_flux(self, split_part=3):
        spectral_flux = self.n_spectral_flux(split_part)
        plt.plot(spectral_flux)
        plt.ylim(top=np.max(spectral_flux) * 1.3)


    # Convert audio frame to time
    def frames_to_time(self, frames):
        return librosa.frames_to_time(frames, n_fft=self.N_FFT, sr=self.SR, hop_length=self.HOP_LEN)


    # Show the plot graph
    def show(self):
        plt.show()