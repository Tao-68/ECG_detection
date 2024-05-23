import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff=40, fs=256, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data

if __name__ == '__main__':
    filepath = "C:/Users/alexa/group_guenther_kosenko_tan_nguyen/lab/eeg_rec.csv"
    sample_freq = 256  # Set the sample frequency accordingly
    cutoff_freq = 40

    # Load the data
    data = pd.read_csv(filepath, header=None)

    # Assuming the data has 4 columns corresponding to the 4 EEG channels
    channels = [data[col] for col in range(4)]

    time = np.arange(data.shape[0]) 

    fig, axes = plt.subplots(4, 2, figsize=(12, 8), sharex=True)

    for i, channel in enumerate(channels):
        # Plot raw data
        axes[i, 0].plot(time, channel)
        axes[i, 0].set_title(f'Raw Data - Channel {i+1}')
        axes[i, 0].set_xlabel('Time [index]')
        axes[i, 0].set_ylabel('Amplitude')

        # Filter the data
        filtered_channel = butter_lowpass_filter(channel, cutoff=cutoff_freq, fs=sample_freq, order=5)

        # Plot filtered data
        axes[i, 1].plot(time, filtered_channel)
        axes[i, 1].set_title(f'Filtered Data (cutoff={cutoff_freq} Hz) - Channel {i+1}')
        axes[i, 1].set_xlabel('Time [index]')
        axes[i, 1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
