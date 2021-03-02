import numpy as np
import tensorflow as tf


def irnv2( input_img ):
    return tf.keras.applications.inception_resnet_v2.preprocess_input( input_img )


def xception( input_img ):
    return tf.keras.applications.xception.preprocess_input( input_img )


def resnet50( input_img ):
    input_img -= 128.8006
    input_img /= 64.6497
    return input_img
