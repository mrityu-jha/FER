import cv2
import dlib
from glob import glob
import pandas as pd
import config
import matplotlib.pyplot as plt
import os 

def make_landmarks_csv( path, csv_path, dat_file ):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor( dat_file )
    images_with_no_face = 0
    images_processed = 0
    df = pd.DataFrame()
    for label in config.CLASS_LABEL:
        files = sorted( glob( os.path.join( path, label.lower() ) + '/*.jpg' ) )        
        for images in files:
            print( images )
            img = cv2.resize(cv2.imread(images),
                             (
                                config.TARGET_SIZE['CV2_LANDMARKS_RESIZE'][0],
                                config.TARGET_SIZE['CV2_LANDMARKS_RESIZE'][1] 
                              )
            )   
            img = cv2.cvtColor( src = img, code = cv2.COLOR_BGR2GRAY )
            faces = detector( img )
            if len( faces ) == 0:
                print( 'Face Not Detected' )
                images_with_no_face += 1
            else:
                row = []
                for face in faces:
                    landmarks = predictor( image = img, box = face )
                    for n in range( 0, 68 ):
                        row.append( landmarks.part( n ).x )
                        row.append( landmarks.part( n ).y )
                    row.append( label )
                    break
                df = df.append( pd.DataFrame( row ).T )
            images_processed += 1
            print("No of Images Processed:", images_processed)
    df.to_csv( csv_path, index = False, header = None )
    print( '\n\nSaving CSV to:', csv_path )
    print('\n\nTotal Images:', images_processed, '\nFace detected in: ',images_processed - images_with_no_face,'images\nFace not detected in: ', images_with_no_face,'images' )
if __name__ == '__main__':
    isTrain = True  # TRUE FOR TRAINING
                    # FALSE FOR TESTING
                    
    if isTrain:
        make_landmarks_csv(
            config.TRAIN_PATH,
            config.TRAIN_FACIAL_LANDMARKS_CSV_PATH,
            config.SHAPE_PREDICTOR_DAT_PATH
        )
    else:
        make_landmarks_csv(
            config.TEST_PATH,
            config.TEST_FACIAL_LANDMARKS_CSV_PATH,
            config.SHAPE_PREDICTOR_DAT_PATH
        )


