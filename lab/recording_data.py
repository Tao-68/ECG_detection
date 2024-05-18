import matplotlib.pyplot as plt
import numpy as np

from utils import setup_stream, connect_stream


BUFFER_LENGTH = 2  # Buffer length in sec
# WINDOW_LENGTH = 0.1  # Length of new chunks in sec
WINDOW_LENGTH = 0.5  # Length of new chunks in sec

# Channel order: TP9, AF7, AF8, TP10


def record_data(inlet, filepath: str, n_buffer: int, n_window: int):
    eeg_buffer = np.zeros((n_buffer, 4))

    try:
        while True:
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=n_window)

            if len(eeg_data) > 0:
                new_data = np.array(eeg_data)[:, :4]
                new_samples = new_data.shape[0]
                eeg_buffer = np.concatenate(
                    [eeg_buffer[new_samples:], new_data], axis=0)

                with open(filepath, 'a') as f:
                    np.savetxt(f, new_data, delimiter=',', fmt='%.5f')

    except KeyboardInterrupt:
        print('Closing!')


def visualize_stream(inlet, n_buffer: int, n_window: int):
    eeg_buffer = np.zeros((n_buffer, 4))

    plt.ion()
    fig, axes = plt.subplots(4)
    lines = []

    for i, ax in enumerate(axes):
        lines.append(ax.plot(eeg_buffer[:, i])[0])

    try:
        while True:
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=n_window)

            if len(eeg_data) > 0:
                new_data = np.array(eeg_data)[:, :4]
                new_samples = new_data.shape[0]
                eeg_buffer = np.concatenate(
                    [eeg_buffer[new_samples:], new_data], axis=0)

                for i, line in enumerate(lines):
                    line.set_ydata(eeg_buffer[:, i])

                for ax in axes:
                    ax.relim()
                    ax.autoscale_view()

                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print('Closing!')


if __name__ == "__main__":
    # filepath = "eeg_rec.csv"
    setup_stream()
    inlet, _ = connect_stream()

    info = inlet.info()
    sample_freq = int(info.nominal_srate())
    n_buffer = int(BUFFER_LENGTH * sample_freq)
    n_window = int(WINDOW_LENGTH * sample_freq)

    # record_data(inlet, filepath, n_buffer, n_window)
    visualize_stream(inlet, n_buffer, n_window)
