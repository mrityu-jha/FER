import tensorflow as tf
import keras 
import os 
import preprocess


ROOT_DIR = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition"

INPUT = os.path.join( ROOT_DIR, 'input' ) 
#INPUT = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/"
TRAIN_PATH = os.path.join( INPUT, 'combined_train' )    
#TRAIN_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/combined_train/"
CSV_PATH = os.path.join( INPUT, 'train.csv' )
#CSV_PATH = os."M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train.csv"
TEST_PATH = os.path.join( INPUT, 'test' )
#TEST_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test/"
IMG_PATH = os.path.join( INPUT, 'train' )
#IMG_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train/"
SAVE_MODEL = os.path.join( INPUT, 'models' )
#SAVE_MODEL = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/models/"

BATCH_SIZE = 64
TARGET_SIZE = ( 197, 197, 3 )
#MODELS = [ 'irnv2', 'xception', 'resnet50' ]
MODELS = [ 'resnet50' ]
NUM_SPLITS = 5
NUM_CLASSES = 7


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
