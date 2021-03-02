import tensorflow as tf
import keras 
import os 
import preprocess


ROOT_DIR = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition"

INPUT = os.path.join( ROOT_DIR, 'input' ) 
#INPUT = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/"
TRAIN_PATH = os.path.join(INPUT, 'train')
#TRAIN_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train"
TRAIN_CSV_PATH = os.path.join( INPUT, 'train.csv' )
#TRAIN_CSV_PATH = os."M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train.csv"
TEST_PATH = os.path.join( INPUT, 'test' )
#TEST_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test"
TEST_CSV_PATH = os.path.join( INPUT, 'test.csv' )
#TEST_CSV_PATH = os."M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test.csv"
SAVE_MODEL = os.path.join( ROOT_DIR, 'models' )
#SAVE_MODEL = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/models"

BATCH_SIZE = 64
TARGET_SIZE = ( 197, 197, 3 )
MODELS = [ 'irnv2', 'xception', 'resnet50' ]
#MODELS = [ 'resnet50' ]
NUM_SPLITS = 5

CLASS_LABEL = {'angry': 0, 'disgust': 1, 'fear': 2,
               'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6, }

NUM_CLASSES = len( CLASS_LABEL )


LAYERS_TO_TRAIN = {
    'irnv2' : 774,
    'xception' : 126,
    'resnet50' : None
}
processing_function = {
    'irnv2' : preprocess.irnv2,
    'xception' : preprocess.xception,
    'resnet50' : preprocess.resnet50
}
