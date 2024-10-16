import tensorflow as tf
import tf2onnx
import onnx
import argparse
from glycan_prediction_model.training.loss import sigmoid_focal_crossentropy


def main(args):
    custom_objects={"sigmoid_focal_crossentropy": sigmoid_focal_crossentropy}

    model = tf.keras.models.load_model(args.i, custom_objects=custom_objects)
    
    input_signature = [tf.TensorSpec([1, 32, 32, 32, 1], tf.float32)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
    onnx.save(onnx_model, args.o)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Keras model path")
    parser.add_argument("-o", help="ONNX model path")
    args = parser.parse_args()
    main(args)