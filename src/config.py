import tensorflow as tf
import keras 


TRAIN_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/combined_train/"
CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train.csv"
TEST_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test/"
INPUT = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/"
IMG_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train/"
SAVE_MODEL = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/models/"

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