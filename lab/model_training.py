import numpy as np
from matplotlib import pyplot as plt
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


def get_dataset(filepath, window_length: int = 128):
    data, labels = _load_file(filepath)

    length = len(data)
    n_windows = length - window_length

    def map_func(idx: int) -> tuple:
        label = tf.reduce_max(labels[idx: idx + window_length])
        label = tf.cast(label, dtype=tf.int8)
        label = tf.one_hot(label, 2)

        indices = tf.range(idx, idx + window_length)
        return tf.gather(data, indices), label

    split_idx = int(0.8 * n_windows)
    train_ds = tf.data.Dataset.range(split_idx).shuffle(n_windows).repeat()
    test_ds = tf.data.Dataset.range(split_idx, n_windows).shuffle(n_windows)
    train_ds = train_ds.map(map_func).batch(32)
    test_ds = test_ds.map(map_func).batch(32)

    return train_ds, test_ds


if __name__ == '__main__':
    # 0. Load model
    model_filepath = "eegnet_model"
    load_model = True
    if load_model:
        model = tf.keras.models.load_model("eegnet_model")

    # 1. Load data
    filepath = "eeg_rec_labeled.csv"

    train_ds, test_ds = get_dataset(filepath)
    # 2. Setup and train model
    if not load_model:
        model = setup_eegnet()
        history = model.fit(train_ds, steps_per_epoch=100, epochs=10, validation_data=test_ds)

    # 3. Calculate confusion matrix (to evaluate model)
    y_true = []
    y_pred = []
    for x, y in test_ds:
        y_true.append(y)
        y_pred.append(model.predict(x))

    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    y_true = tf.argmax(y_true, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    # save model
    if not load_model:
        model.save(model_filepath, save_format='tf')

    # plot the loss and rmse
        for metric, result in history.history.items():
            if 'loss' in metric:
                result = tf.sqrt(result)
            plt.plot(result, label=metric.replace('loss', 'rmse'))
        plt.legend()
        plt.show()



