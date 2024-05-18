import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from utils import setup_eegnet


def _load_file(filepath: str) -> tuple:
    data = tf.io.read_file(filepath)
    data = tf.strings.split(data, sep='\r\n')
    data = tf.strings.split(data, sep=',')[1:-1].to_tensor()

    data = tf.strings.to_number(data)
    data, labels = data[:, :-1], data[:, -1]

    return data, labels


def get_dataset(filepath, window_length: int = 125):
    data, labels = _load_file(filepath)

    ...

    return train_ds, test_ds


if __name__ == '__main__':
    # 1. Load data
    # 2. Setup and train model
    # 3. Calculate confusion matrix (to evaluate model)

    print(cm)
    model.save(...)
