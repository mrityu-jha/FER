import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import os
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, StratifiedKFold
import config
import model_dispatcher
import plot

def return_split( Y ):
    skf = StratifiedKFold( n_splits = config.NUM_SPLITS, shuffle = True, random_state = 42 )
    split = 1
    for train_idx, val_idx in skf.split( np.zeros( Y.shape[0] ), Y ):
        print( 'Returning Split:', split )
        split += 1
        yield train_idx, val_idx


def return_gen( name_of_model ):
    print( 'Returning Generators for:', name_of_model )
    train_datagen = ImageDataGenerator(
        preprocessing_function = config.processing_function[ name_of_model ]
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function = config.processing_function[ name_of_model ]
    )

    return train_datagen, val_datagen

def get_model_name( num_fold, isBest = False ):
    print( 'Num-Fold:', num_fold, 'Model Name:', config.MODELS[ num_fold - 1 ], 'isBest:', str( isBest ) )
    if isBest:
        return os.path.join( config.SAVE_MODEL, 'best_' + config.MODELS[ num_fold - 1 ] + '_' + str( num_fold ) + '.h5' )
    else:
        return os.path.join( config.SAVE_MODEL, config.MODELS[num_fold - 1] + '_' + str( num_fold ) + '.h5' )


def return_callbacks( num_fold, isBest = False ):
    print( 'Returning Callbacks' )
    mc = ModelCheckpoint(
        filepath = get_model_name( num_fold, isBest ),
        monitor = 'val_accuracy',
        mode = 'max',
        verbose = 1,
        save_best_only = True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_accuracy',
        mode = 'max',
        factor = 0.5,
        patience = 3,
        verbose = 1
    )

    return mc, reduce_lr

def return_opt( opt_name, learning_rate ):
    print( opt_name.upper(), 'Optimizer Selected with learning rate:', learning_rate )
    opt_dict = {
        'adam' : tf.keras.optimizers.Adam( learning_rate = learning_rate ),
        'sgd'  : tf.keras.optimizers.SGD( learning_rate = learning_rate ),
        'rms'  : tf.keras.optimizers.RMSprop( learning_rate = learning_rate ),
        'ftrl' : tf.keras.optimizers.Adagrad( learning_rate = learning_rate ),
        'nadam': tf.keras.optimizers.Nadam( learning_rate = learning_rate ),
        'ada_delta' : tf.keras.optimizers.Adadelta( learning_rate = learning_rate ),
        'ada_grad'  : tf.keras.optimizers.Adagrad( learning_rate = learning_rate ),
        'ada_max'   : tf.keras.optimizers.Adamax( learning_rate = learning_rate ),
    }

    return opt_dict[ opt_name ]


def train():
    print( 'Starting Training' )
    data_frame = pd.read_csv( config.CSV_PATH, dtype = str )
    Y = data_frame[['Label']].copy()
    num_fold = 1
    data = dict()

    for train_idx, val_idx in return_split( Y ):
        train_df = data_frame.iloc[train_idx]
        val_df = data_frame.iloc[val_idx]
        name_of_model = config.MODELS[ num_fold - 1 ]
        train_datagen, val_datagen = return_gen( name_of_model )

        train_data = train_datagen.flow_from_dataframe(
            dataframe = train_df,
            directory = config.TRAIN_PATH,
            x_col = "Image",
            y_col = "Label",
            target_size = ( config.TARGET_SIZE[0], config.TARGET_SIZE[1] ),
            class_mode = "categorical",
            shuffle = True,
            batch_size = config.BATCH_SIZE,
            seed = 42
         )

        val_data = val_datagen.flow_from_dataframe(
            dataframe = val_df,
            directory = config.TRAIN_PATH,
            x_col = "Image",
            y_col = "Label",
            target_size = ( config.TARGET_SIZE[0], config.TARGET_SIZE[1] ),
            class_mode = "categorical",
            shuffle = True,
            batch_size = config.BATCH_SIZE,
            seed = 42
        )

        model = model_dispatcher.return_model( name_of_model )
        mc, reduce_lr = return_callbacks(
            num_fold,
            False
        )

        opt = return_opt( 'adam', learning_rate = 0.01 )
        model.compile(
            optimizer = opt,
            loss = 'categorical_crossentropy',
            metrics = ['accuracy']
        )

        history = model.fit(
            train_data,
            validation_data = val_data,
            epochs = 1,
            callbacks = [ mc, reduce_lr ],
            steps_per_epoch = train_data.__len__()
        )

        plot.plot_loss( history )
        plot.plot_accuracy( history )
        model.load_weights( get_model_name(num_fold) )
        results = model.evaluate( val_data )
        results = dict( zip( model.metrics_name, results ) )

        data[num_fold] = [ train_data, val_data ]
        num_fold += 1
        tf.keras.backend.clear_session()

        if( num_fold >= 3 ):
            break

    return data



def fine_tune( num_fold, data ):
    #InceptionResNetV2 - 774:
    #Xception - 126
    print( 'Fine-Tuning' )
    model = None
    train_data = None
    val_data = None

    train_data, val_data = data[num_fold - 1][0], data[num_fold - 1][1]
    model = model_dispatcher.return_model(config.MODELS[num_fold - 1])

    model.load_weights(get_model_name(num_fold))
    #LAYERS_TO_TRAIN = some_arbitrary_value
    print( model.summary() )
    for layers in model.layers[1].layers[config.LAYERS_TO_TRAIN[name_of_model]:]:
        layers.trainable = True
    print( model.summary() )

    mc, reduce_lr = return_callbacks(
        num_fold,
        True
    )

    opt = return_opt( 'adam', learning_rate = 1e-5 )
    model.compile(
        optimizer = opt,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data = val_data,
        epochs = 20,
        callbacks = [ mc, reduce_lr ],
        steps_per_epoch = train_data.__len__()
    )
    plot.plot_loss( history )
    plot.plot_accuracy( history )



if __name__ == '__main__':
    data = train()
    num_fold = 1
    for name_of_model in config.MODELS:
        fine_tune( num_fold, data )
        num_fold += 1
