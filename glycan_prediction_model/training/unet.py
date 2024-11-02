"U-Net - Original Implementation Paul Bond, Edited by Jordan Dialpuri"

import tensorflow as tf

_ARGS = {"padding": "same", "activation": "relu", "kernel_initializer": "he_normal"}
_downsampling_args = {
    "padding": "same",
    "use_bias": False,
    "kernel_size": 3,
    "strides": 1,
}


def binary_model():
    inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))
    skip_list = []

    filter_list = [16, 32, 64, 128, 256]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # outputs = tf.keras.layers.Conv3D(1, 3, padding="same", activation="sigmoid")(x)
    outputs = tf.keras.layers.Conv3D(2, 1, padding="same", activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)


def binary_model2():
    inputs = x = tf.keras.Input(shape=(32, 32, 32, 1))
    skip_list = []

    filter_list = [16, 32, 64, 128, 256]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv3D(2, 1, padding="same", activation="softmax")(x)
    # outputs = tf.keras.layers.Dense(2, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)

def binary_model_64():
    inputs = x = tf.keras.Input(shape=(64, 64, 64, 2))
    skip_list = []

    filter_list = [16, 32, 64, 128, 256]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        x = tf.keras.layers.GroupNormalization(filters)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv3D(2, 1, padding="same", activation="softmax")(x)
    # outputs = tf.keras.layers.Dense(2, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def binary_model_64_2():
    inputs = x = tf.keras.Input(shape=(64, 64, 64, 2))
    skip_list = []

    filter_list = [16, 32, 64, 128, 256]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    outputs = tf.keras.layers.Conv3D(2, 1, padding="same", activation="softmax")(x)
    # outputs = tf.keras.layers.Dense(2, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def binary_model3():
    inputs = x = tf.keras.Input(shape=(64, 64, 64, 1))
    skip_list = []

    filter_list = [16, 32, 64, 128, 256, 512]

    for filters in filter_list:
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, **_downsampling_args)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        skip_list.append(x)
        x = tf.keras.layers.MaxPool3D(2)(x)
        # x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)
    x = tf.keras.layers.Conv3D(512, 3, **_ARGS)(x)

    for filters in reversed(filter_list):
        x = tf.keras.layers.Conv3DTranspose(filters, 3, 2, padding="same")(x)
        x = tf.keras.layers.concatenate([x, skip_list.pop()])
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv3D(filters, 3, **_ARGS)(x)
        # x = tf.keras.layers.GroupNormalization(filters)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv3D(1, 1, padding="same", activation="sigmoid")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    binary_model2().summary()
