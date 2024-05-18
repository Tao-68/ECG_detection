from utils import setup_stream, connect_stream


def prediction_stream(inlet,
                      model_path: str,
                      window_length: int,
                      n_buffer: int,
                      n_update: int):
    pass


if __name__ == "__main__":
    # Setup stream
    setup_stream()
    inlet, _ = connect_stream()

    # Use stream for live predictions
    prediction_stream(...)
