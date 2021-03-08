import pandas as pd
import numpy as np
from tensorflow.python.autograph.impl.api import convert
import plot
import preprocess
import config
from train import return_callbacks, return_opt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import model_dispatcher
import preprocess

def return_split( X, Y ):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.25,
        shuffle=True,
        stratify=Y
    )

    print( 'X:', X.shape, '\nX_train:', X_train.shape, '\nY', Y.shape, '\nY_train:', Y_train.shape, '\nX_test:', X_test.shape, '\nY_test:', Y_test.shape )
    return X_train, X_test, Y_train, Y_test

def conver_to_ohe( *arrays ):
    converted_arrays = []
    for arr in arrays:
        arr = to_categorical( arr )
        converted_arrays.append( arr )
    return converted_arrays

def train( num_fold ):
    isBest = True
    data_frame = pd.read_csv( 
        config.TRAIN_FACIAL_LANDMARKS_CSV_PATH,
        header = None
    )

    data_frame = preprocess.customFacialLandmarks( data_frame, config.CLASS_LABEL )

    X = data_frame.iloc[:, :-1].copy().values
    Y = data_frame.iloc[:, -1].copy().values.reshape( -1, 1 )   
    X_train, X_test, Y_train, Y_test = return_split( X, Y )

    model = model_dispatcher.return_model( num_fold )
    Y_train_enc, Y_test_enc = conver_to_ohe( Y_train, Y_test )
    print( Y_train_enc.shape, Y_test_enc.shape )

    mc, reduce_lr = return_callbacks(
        num_fold,
        isBest
    )

    opt = return_opt('adam', learning_rate=0.01)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit( 
        X_train,
        Y_train_enc,
        epochs = 100,
        validation_data = ( X_test, Y_test_enc ),
        batch_size = config.BATCH_SIZE,
        callbacks = [ mc, reduce_lr ]
    )

    plot.plot_loss(history)
    plot.plot_accuracy(history)



if __name__ == '__main__':
    for num_fold in config.LIST_OF_FOLD_EXCEPTIONS:
        train( num_fold )
    
    
