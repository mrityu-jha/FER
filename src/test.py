import config
import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import evaluation
import pandas as pd
import test_facialLandmarks


def return_gen( path ):
    name_of_model = path.split('_')[1]
    print('Returning Validation Generator for:', name_of_model )
    test_datagen = ImageDataGenerator(
        preprocessing_function=config.preprocessing_function[name_of_model]
    )
    return test_datagen

def predict_separate( path, test_df ):
    print( "Loading Model from: ", path )
    name_of_model = path.split('_')[1]
    model = load_model( path )
    test_datagen = return_gen(path)
    test_data = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = None,
        x_col = "Image",
        y_col = "Label",
        target_size=(config.TARGET_SIZE[name_of_model][0], config.TARGET_SIZE[name_of_model][1]),
        class_mode = "categorical",
        batch_size = config.BATCH_SIZE,
        seed = 42,
        shuffle = False
    )
    y_predProbs = model.predict(test_data, steps = test_data.__len__(), verbose=1)
    if name_of_model == 'resnet50' :
        y_predProbs[ :, :] = y_predProbs[ :, [ 0, 1, 2, 3, 6, 4, 5 ] ]
    
    return y_predProbs

def predict_ensemble():
    y_predProbs = np.zeros( 1, dtype = float )
    test_df = pd.read_csv( config.TEST_CSV_PATH, dtype = str )
    for path in os.listdir( config.SAVE_MODEL ):
        if os.path.isdir( os.path.join( config.SAVE_MODEL, path ) ):
            if 'custom' in os.path.join( config.SAVE_MODEL, path ):
                y_predProbs = y_predProbs + test_facialLandmarks.predict_separate(
                    os.path.join(config.SAVE_MODEL, path)
                )[0]
            else:
                y_predProbs = y_predProbs + predict_separate(
                    os.path.join(config.SAVE_MODEL, path),
                    test_df 
                )

    y_true = test_df[['Label']].copy()
    y_pred = np.argmax( y_predProbs, axis=-1 )
    class_label = dict( ( v, k ) for k, v in config.CLASS_LABEL.items() )

    y_pred = [class_label[k] for k in y_pred]
    evaluation.accuracy_of_model( y_true, y_pred )
    evaluation.confusionMatrix_of_model( y_true, y_pred )
    evaluation.classificationReport_of_model( y_true, y_pred )

if __name__ == '__main__':
    predict_ensemble()
