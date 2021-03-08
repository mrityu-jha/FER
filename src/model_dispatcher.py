from os import name
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, GlobalAveragePooling2D
from keras.applications import Xception, InceptionResNetV2
from keras_vggface.vggface import VGGFace
from keras.models import load_model
from tensorflow.python.ops.gen_math_ops import xlog1py
import config


def irnv2( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    base_model = InceptionResNetV2( include_top = False, pooling = 'avg', input_tensor = inputs )
    for layers in base_model.layers:
        layers.trainable = False
    x = base_model( base_model.inputs )
    x = Dropout( 0.5 )( x )
    x = Dense( 128, activation = 'relu' )( x )
    outputs = Dense( config.NUM_CLASSES, activation = 'softmax' )( x )
    return keras.Model( inputs, outputs )

def xception( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    base_model = Xception( include_top = False, pooling = 'avg' )
    for layers in base_model.layers:
        layers.trainable = False
    x = base_model( inputs )
    x = Dropout( 0.5 )( x )
    x = Dense( 128, activation = 'relu' )( x )
    outputs = Dense( config.NUM_CLASSES, activation='softmax')(x)
    return keras.Model( inputs, outputs )

def resnet50( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    base_model = VGGFace( model = 'resnet50', weights = 'vggface', include_top = False, input_tensor = inputs )
    for layers in base_model.layers:
        layers.trainable = False
    x = base_model.output
    x = Flatten()( x )
    x = Dense( 1024, activation = 'relu' )( x )
    outputs = Dense( config.NUM_CLASSES, activation='softmax')(x)
    return keras.Model( inputs, outputs )

def customFacialLandmarks( TARGET_SIZE ):
    inputs = tf.keras.Input( shape = TARGET_SIZE )
    x = BatchNormalization()( inputs )
    x = Dense(1024, activation='relu')(x)
    outputs = Dense( config.NUM_CLASSES, activation = 'softmax' )( x )
    return keras.Model( inputs, outputs )

def return_model( num_fold = None, name_of_model = None ):
    if num_fold is not None and name_of_model is None:
        print( 'NUM FOLD: ', num_fold )
    elif num_fold is None and name_of_model is not None:
        print( 'Name of Model: ', name_of_model )
    elif num_fold is not None and name_of_model is not None:
        print( 'Atleast one of the arguments passed must be None')
        return None
    else:
        print( 'Atleast one of the arguments passed must be not None')
        return None

    model_dict = { 
        'irnv2' : irnv2( config.TARGET_SIZE['irnv2'] ),
        'xception' : xception( config.TARGET_SIZE['xception'] ),
        'resnet50' : resnet50( config.TARGET_SIZE['resnet50'] ),
        'customFacialLandmarks' : customFacialLandmarks( config.TARGET_SIZE['customFacialLandmarks'] )
    }
    if( num_fold == None ):
        try:
            return model_dict[name_of_model]
        except:
            print( 'Invalid Model Name passed' )
    else:
        try:
            print( 'MODEL SELECTED: ', config.MODELS[ num_fold - 1  ] )
            return model_dict[ config.MODELS[ num_fold - 1 ] ]
        except:
            print( 'The value of num_fold is invalid' )
            
