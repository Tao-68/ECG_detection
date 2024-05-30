from multiprocessing import Process
from time import sleep

from pylsl import StreamInlet, resolve_byprop
from muselsl import stream, list_muses
import tensorflow as tf

def add_labels(filepath_rec: str = "eeg_rec.csv", filepath_labels: str = "labels.csv"):
    # load labels into array of tupels
    labels = []
    with open(filepath_labels, 'r') as f:
        for line in f:
            labels.append(line.strip().split(','))
    lines_of_blink = []
    for blink in labels:
        start, end = blink
        start = int(start)
        end = int(end)
        lines_of_blink.extend(list(range(start, end)))
    # load eeg data
    data = []
    with open(filepath_rec, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    # append labels to eeg data
    for i, line in enumerate(data):
        if i in lines_of_blink:
            data[i].append(1)
        else:
            data[i].append(0)
    # save data
    filepath_rec_labeled = filepath_rec.split('.')[0]
    filepath_rec_labeled += "_labeled.csv"
    with open(filepath_rec_labeled, 'w') as f:
        for line in data:
            line = [str(x) for x in line]
            f.write(','.join(line) + '\n')


def connect_stream():
    print('Looking for an EEG stream...')

    for attempt in range(20):
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            print(f"No stream found - attempt: {attempt}")
            sleep(1)

        else:
            print("Start acquiring data")
            inlet = StreamInlet(streams[0], max_chunklen=12)
            eeg_time_correction = inlet.time_correction()
            break
    else:
        raise RuntimeError('Can\'t find EEG stream.')

    return inlet, eeg_time_correction


def setup_stream(address: str = None):
    muses = list_muses()

    for muse in muses:
        if muse['name'] == 'MuseS-8FDB':
            process = Process(target=stream, args=(muse['address'],))
            process.start()
            break
    else:
        raise ConnectionError('No Muses found')


def setup_eegnet(n_channels: int = 4, window_length: int = 128):

    kernel_length = window_length // 2

    input1 = tf.keras.layers.Input(shape=(window_length, n_channels))
    permute = tf.keras.layers.Permute((2, 1))(input1)
    expand = tf.keras.layers.Reshape((n_channels, window_length, 1))(permute)

    block1 = tf.keras.layers.Conv2D(
        8, (1, kernel_length), padding='same', use_bias=False)(expand)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.DepthwiseConv2D(
        (n_channels, 1), use_bias=False, depth_multiplier=2,
        depthwise_constraint=tf.keras.constraints.max_norm(1.))(block1)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.Activation('elu')(block1)
    block1 = tf.keras.layers.AveragePooling2D((1, 4))(block1)
    block1 = tf.keras.layers.Dropout(0.5)(block1)

    block2 = tf.keras.layers.SeparableConv2D(
        16, (1, 16), use_bias=False, padding='same')(block1)
    block2 = tf.keras.layers.BatchNormalization()(block2)
    block2 = tf.keras.layers.Activation('elu')(block2)
    block2 = tf.keras.layers.AveragePooling2D((1, 8))(block2)
    block2 = tf.keras.layers.Dropout(0.5)(block2)

    flatten = tf.keras.layers.Flatten()(block2)

    dense = tf.keras.layers.Dense(
        2, kernel_constraint=tf.keras.constraints.max_norm(0.25))(flatten)
    softmax = tf.keras.layers.Activation('softmax')(dense)

    model = tf.keras.models.Model(inputs=input1, outputs=softmax)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    add_labels()