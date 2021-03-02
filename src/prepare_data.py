import pandas as pd
import os
import glob
import config


def create_csv( path_of_root, csv_path ):
    image_path = []
    image_label = []
    for label in config.CLASS_LABEL:  
        files = glob.glob( os.path.join( path_of_root, label ) + '/*.jpg')
        for names in files:
            image_path.append(names)
            image_label.append(label.capitalize())

    df = pd.DataFrame(list(zip(image_path, image_label)), columns=[
                         'Image', 'Label']).sample(frac=1).reset_index(drop=True)
    print( "Head of the DataFrame: ", df.head() )
    print( "No of Files Found: ", len(image_path))

    df.to_csv(
        csv_path,
        index = False
    )



if __name__ == '__main__':
    isTrain = False   #TRUE FOR TRAINING 
                     #FALSE FOR TESTING


    if isTrain:
        create_csv( 
            config.TRAIN_PATH,
            config.TRAIN_CSV_PATH
        )
    else:
        create_csv( 
            config.TEST_PATH,
            config.TEST_CSV_PATH
        )

