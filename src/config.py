import tensorflow as tf
import keras 
import os 

ROOT_DIR = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition"
INPUT = os.path.join( ROOT_DIR, 'input' ) 
#"M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/"
TRAIN_PATH = os.path.join( INPUT, 'combined_train' )    
#"M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/combined_train/"
CSV_PATH = os.path.join( INPUT, 'train.csv' )
#CSV_PATH = os."M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train.csv"
TEST_PATH = os.path.join( INPUT, 'test' )
#TEST_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test/"
IMG_PATH = os.path.join( INPUT, 'train' )
#IMG_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train/"
SAVE_MODEL = os.path.join( INPUT, 'models' )
#SAVE_MODEL = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/models/"

BATCH_SIZE = 64
TARGET_SIZE = ( 224, 224, 3 )
MODELS = [ 'irnv2', 'xception' ]
NUM_SPLITS = 5

LAYERS_TO_TRAIN = {
    'irnv2' : 774,
    'xception' : 126
}
processing_function = {
    MODELS[0] : tf.keras.applications.inception_resnet_v2.preprocess_input,
    MODELS[1] : tf.keras.applications.xception.preprocess_input
}
