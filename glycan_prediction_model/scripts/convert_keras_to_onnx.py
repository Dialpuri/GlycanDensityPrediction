import argparse
import numpy as np
import tensorflow as tf
import keras


def sigmoid_focal_crossentropy(
    y_true, y_pred, alpha=0.25, gamma=2.0, from_logits: bool = False
):
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)


def main(args):
    model = tf.keras.models.load_model(args.i, custom_objects={ 
                    "sigmoid_focal_crossentropy": sigmoid_focal_crossentropy, 
                })
    model.export("models/saved_model")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to keras model")
    parser.add_argument("-o", help="Path to onnx model", default="model.onnx")
    main(parser.parse_args())
    
    # python -m tf2onnx.convert --saved-model models/saved_model/ --output models/model.onnx

