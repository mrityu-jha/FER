import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.applications import Xception, InceptionResNetV2
from keras.preprocessing import image
import config


def irnv2( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    base_model = InceptionResNetV2( include_top = False, pooling = 'avg', input_tensor = inputs )
    for layers in base_model.layers:
        layers.trainable = False
    x = base_model( base_model.inputs, training = False )
    x = Dropout( 0.5 )( x )
    x = Dense( 128, activation = 'relu' )( x )
    outputs = Dense( 6, activation = 'softmax' )( x )
    return keras.Model( inputs, outputs )

def xception( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    base_model = Xception( include_top = False, pooling = 'avg', input_tensor = inputs )
    for layers in base_model.layers:
        layers.trainable = False
    x = base_model( base_model.inputs, training = False )
    x = Dropout( 0.5 )( x )
    x = Dense( 128, activation = 'relu' )( x )
    outputs = Dense( 6, activation = 'softmax' )( x )
    return keras.Model( inputs, outputs )

# def model_dict( TARGET_SIZE = config.TARGET_SIZE ):
#     model_dict = { 'irnv2' : irnv2( TARGET_SIZE ), 'xception' : xception( TARGET_SIZE ) }
#     return model_dict


def return_model( name_of_model ):
    if name_of_model == 'xception':
        return xception( config.TARGET_SIZE )
    
    elif name_of_model == 'irnv2':
        return irnv2( config.TARGET_SIZE )

    else:
        print( 'Invalid Model Name' )

    
