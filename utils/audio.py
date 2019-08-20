import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import math


class Audio:
    def __init__(self, y, sr, n_fft, hop_len):
        self.y = y
        self.SR = sr
        self.N_FFT = n_fft
        self.HOP_LEN = hop_len
        self.S = np.abs(librosa.core.stft(y, n_fft=self.N_FFT, hop_length=self.HOP_LEN))
        self.Sdb = librosa.amplitude_to_db(self.S, ref=1.0)
        self.freqs = librosa.core.fft_frequencies(n_fft=n_fft)


    # Remove background noise
    def remove_noise(self, Sdb, ratio=.5, split_part=100):
        part_size = math.ceil(Sdb.shape[1] / split_part)
        sdb_with_mean = []
        # Split Sdb into multiple part and get each part's mean dB.
        for idx in range(0, Sdb.shape[1] - 1, part_size):
            part = Sdb[:, idx:idx + part_size]
            part_mean = np.mean(part)
            sdb_with_mean.append((part, part_mean))
        # Sort the mean dB
        sdb_with_mean.sort(key=lambda tup: tup[1])

        # Get the mean dB which is smaller than most other part
        # and assume it is background noise
        background = []
        for idx in range(math.ceil(split_part * ratio)):
            background.append(sdb_with_mean[idx][0])

        # Use background noise to normalize the whole audio file.
        # (to remove most part of the background noise from the file)
        background = np.concatenate(background, axis=1)
        Sdb = self.normalize_with_part(Sdb, background)
        Sdb -= .5  # Make most of the noise get down below of 0
        Sdb[Sdb <= 0] = 0  # Wipe out noise

        return Sdb


    # Normalization only with part of the x
    def normalize_with_part(self, x, part_x):
        part_x_median = np.median(part_x)
        part_x -= part_x_median
        part_x_std = np.std(part_x)

        x -= part_x_median
        x /= part_x_std

        return x


    # Split Sdb into multiple part
    def split_sdb(self, split_part=10):
        Sdb = self.Sdb.copy()
        part_size = math.ceil(len(self.freqs) / split_part)

        sdb_list = []
        for idx in range(part_size, len(self.freqs), part_size):
            sdb_list.append(Sdb[idx - part_size:idx])
        sdb_list.append(Sdb[idx:])

        return sdb_list


    # Plot the Spectrogram
    def spectrogram(self):
        plt.figure(figsize=(10, 4))
        plt.title("Mel Spectrogram")
        librosa.display.specshow(self.Sdb, y_axis='mel', x_axis='time', sr=self.SR, hop_length=self.HOP_LEN)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()


    # Draw vertical lines on the plot
    def draw_vlines(self, frames, x='time', color='w'):
        if (x == 'time'):
            y = librosa.frames_to_time(frames, sr=self.SR, hop_length=self.HOP_LEN)
        elif (x == 'frame'):
            y = frames
        else:
            assert ("Unknown value for parameter X")

        plt.vlines(y, 0, self.freqs[-1], colors=color, linestyles='--')


    # Convert audio frame to time
    def frames_to_time(self, frames):
        return librosa.frames_to_time(frames, n_fft=self.N_FFT, sr=self.SR, hop_length=self.HOP_LEN)


    # Show the plot graph
    def show(self):
        plt.show()