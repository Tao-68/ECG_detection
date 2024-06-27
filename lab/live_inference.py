from utils import setup_stream, connect_stream
from recording_data import record_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def prediction_stream(inlet,
                      model_path: str,
                      window_length: int,
                      n_buffer: int,
                      n_update: int):

    eeg_buffer = np.zeros((n_buffer, 4))

    try:
        while True:
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=n_update)

            if len(eeg_data) > 0:
                new_data = np.array(eeg_data)[:, :4]
                new_samples = new_data.shape[0]
                eeg_buffer = np.concatenate(
                    [eeg_buffer[new_samples:], new_data], axis=0)

                # Predict
                eeg_window = eeg_buffer[-window_length:]
                eeg_window = np.expand_dims(eeg_window, axis=0)
                prediction = model.predict(eeg_window, verbose=0)
                # live plot of prediction
                


    except KeyboardInterrupt:
        print('Closing!')
    pass

if __name__ == "__main__":
    # Setup stream
    BUFFER_LENGTH = 2  # Buffer length in sec
    WINDOW_LENGTH = 0.5  # Length of new chunks in sec
    setup_stream()
    inlet, _ = connect_stream()
    info = inlet.info()
    sample_freq = int(info.nominal_srate())
    n_buffer = int(BUFFER_LENGTH * sample_freq)
    n_window = int(WINDOW_LENGTH * sample_freq)

    # Load model
    model_filepath = "eegnet_model"
    model = tf.keras.models.load_model("eegnet_model")

    # Use stream for live predictions
    prediction_stream(inlet, model_filepath, n_window, n_buffer, 10)
