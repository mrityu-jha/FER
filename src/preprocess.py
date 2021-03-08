import numpy as np
import tensorflow as tf
import pandas as pd

def irnv2( input_img ):
    return tf.keras.applications.inception_resnet_v2.preprocess_input( input_img )


def xception( input_img ):
    return tf.keras.applications.xception.preprocess_input( input_img )


def resnet50( input_img ):
    input_img -= 128.8006
    input_img /= 64.6497
    return input_img


def customFacialLandmarks( data_frame, CLASS_LABEL ):
    data_frame.iloc[:,-1] = data_frame.iloc[:,-1].apply( lambda x : CLASS_LABEL[x.capitalize()] )
    return data_frame
