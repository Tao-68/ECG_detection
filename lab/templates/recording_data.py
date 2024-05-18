from utils import setup_stream, connect_stream


def record_data(inlet, filepath: str, n_window: int):
    try:
        while True:
            # Pull data from device
            # Write (new) data to file
            # Update visualization (with data in buffer)

    except KeyboardInterrupt:
        print('Closing!')


def visualize_stream(inlet, n_buffer: int, n_window: int):
    try:
        while True:
            # Pull data from device
            # Update buffer
            # Update visualization (with data in buffer)

    except KeyboardInterrupt:
        print('Closing!')


if __name__ == "__main__":
    # Setup EEG stream and connection
    setup_stream()
    inlet, _ = connect_stream()

    # Record or visualize data
    record_data(inlet, filepath, n_window)
    visualize_stream(inlet, n_buffer, n_window)
