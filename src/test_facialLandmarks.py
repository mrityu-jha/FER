import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.saving.save import load_model
import glob
import config
import cv2
import dlib 
import os
import evaluation
from glob import glob

def predict_separate(path, test_path=config.TEST_PATH, dat_file=config.SHAPE_PREDICTOR_DAT_PATH):
    print( 'Loading Model from: ', path )
    name_of_model = path.split( '_' )[1]
    model = load_model( path )
    images_processed = 0
    images_with_no_face = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dat_file)
    y_predProbs = np.empty( ( 0, 7 ) )
    y_true = []
    for label in config.CLASS_LABEL:
        files = sorted( glob( os.path.join( test_path, label.lower() ) + '/*.jpg' ) )        
        for images in files:
            img = cv2.resize(cv2.imread(images),
                            (
                                config.TARGET_SIZE['CV2_LANDMARKS_RESIZE'][0],
                                config.TARGET_SIZE['CV2_LANDMARKS_RESIZE'][1]
                            )
            )
            img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
            faces = detector(img)
            if len(faces) == 0:
                images_with_no_face += 1
                y_predProbs = np.vstack( 
                    (y_predProbs, 
                    np.zeros( config.NUM_CLASSES ))        
            )
            else:
                row = []
                for face in faces:
                    landmarks = predictor(image=img, box=face)
                    for n in range(0, 68):
                        row.append(landmarks.part(n).x)
                        row.append(landmarks.part(n).y)
                    break
                y_predProbs = np.vstack(
                    (y_predProbs,
                    model.predict( np.array( row ).reshape( 1, -1 ), verbose = 1 ))
                )
            y_true.append( label )
            images_processed += 1 
            print("No of Images Processed:", images_processed )
    print('\n\nTotal Images:', images_processed, '\nFace detected in: ', images_processed -
          images_with_no_face, 'images\nFace not detected in: ', images_with_no_face, 'images')
    return y_predProbs, y_true
                    
def predict_ensemble():
    y_predProbs = np.zeros(1, dtype=float)
    for path in os.listdir(config.SAVE_MODEL):
        if os.path.isdir(os.path.join(config.SAVE_MODEL, path)):
            if 'custom' in os.path.join(config.SAVE_MODEL, path):
                temp, y_true = predict_separate(
                    os.path.join(config.SAVE_MODEL, path)
                )
                y_predProbs = y_predProbs + temp

    y_pred = np.argmax(y_predProbs, axis=-1)
    class_label = dict((v, k) for k, v in config.CLASS_LABEL.items())
    y_pred = [class_label[k] for k in y_pred]
    evaluation.accuracy_of_model(y_true, y_pred)
    evaluation.confusionMatrix_of_model(y_true, y_pred)
    evaluation.classificationReport_of_model(y_true, y_pred)

if __name__ == '__main__':
    predict_ensemble()
