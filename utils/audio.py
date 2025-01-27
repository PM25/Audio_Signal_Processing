import numpy as np
import librosa.display
from librosa import time_to_frames
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math


class Audio():
    def __init__(self, y, sr, n_fft, hop_len):
        self.y = y
        self.SR = sr
        self.N_FFT = n_fft
        self.HOP_LEN = hop_len
        self.S = np.abs(librosa.core.stft(y, n_fft=self.N_FFT, hop_length=self.HOP_LEN))
        self.Sdb = librosa.amplitude_to_db(self.S, ref=1.0)
        self.avg_db = self.avg_db_timeline(self.Sdb)
        self.freqs = librosa.core.fft_frequencies(n_fft=n_fft)


    # Onset & Offset Detection
    def onset_offset_detect(self, flux_threshold=1, db_threshold=-20, seg_threshold=0.05, split_part=10, epsilon=.1):
        potential_onsets_offsets = []

        for Sdb in self.split_sdb(split_part):
            clean_Sdb = self.remove_noise(Sdb)  # Wipe out background noise
            spectral_flux = self.spectral_flux(clean_Sdb)  # Sum up the spectral flux
            potential_onsets_offsets += self.find_start_end_peaks(spectral_flux, flux_threshold=flux_threshold, db_threshold=db_threshold, epsilon=epsilon)

        onsets, offsets = [], []
        for group in self.segmentation(potential_onsets_offsets, seg_threshold):
            if (len(group) >= split_part / 2):
                onset, offset = group[0]
                onsets.append(onset)
                offsets.append(offset)

        return (onsets, offsets)


    # Apply simple segmentation on 1D tuple(onset, offset) array
    # threshold means how much difference below can be seen as same onsets or offsets
    def segmentation(self, array, threshold):
        threshold = time_to_frames(threshold, sr=self.SR, hop_length=self.HOP_LEN, n_fft=self.N_FFT)
        if(threshold < 1):
            threshold = 1

        results, group = [], []
        for (begin, end) in sorted(array, key=lambda tup: tup[0]):
            if (group == []):
                group.append((begin, end))
            else:
                if (begin <= group[0][0] + threshold):
                    group.append((begin, end))
                else:
                    results.append(group.copy())
                    group.clear()
                    group.append((begin, end))
        results.append(group)

        return results


    # Find all the peaks by threshold, then find the beginning of it ( < epsilon)
    def find_start_end_peaks(self, data, flux_threshold=1, db_threshold=-10, epsilon=.1):
        peaks, _ = find_peaks(data, height=flux_threshold)
        start_end = []
        for peak in peaks:
            start, end = None, None
            # Find start of the peaks
            for idx in range(peak, 0, -1):
                if (data[idx] <= epsilon and self.avg_db[idx] <= db_threshold):
                    start = idx
                    break
            # Find Negative Flux
            for idx in range(peak, len(data)):
                if (data[idx] < -epsilon):
                    neg_flux_start = idx
                    break
            # Find end of the peaks
            for idx in range(neg_flux_start, len(data)):
                if (data[idx] <= epsilon and self.avg_db[idx] < db_threshold):
                    end = idx
                    break
            if(start != None and end != None):
                start_end.append((start, end))

        return start_end


    # Sum the positive change in each frequency bin
    def spectral_flux(self, Sdb):
        return  np.mean(np.diff(Sdb, axis=1), axis=0)


    # Remove background noise
    def remove_noise(self, Sdb, ratio=.7, split_part=100):
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
        part_x_mean = np.mean(part_x)
        part_x -= part_x_mean
        part_x_std = np.std(part_x)

        x -= part_x_mean
        if(part_x_std > 1):
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


    # Mean dB of all frequecy range vs Time
    def avg_db_timeline(self, Sdb):
        return [np.mean(Sdb[:, idx]) for idx in range(Sdb.shape[1])]


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