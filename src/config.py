import os
import preprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split( 'src' )[0]
#ROOT_DIR = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition"
INPUT = os.path.join( ROOT_DIR, 'input' )
#INPUT = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input"
TRAIN_PATH = os.path.join(INPUT, 'train')
#TRAIN_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/train"
TEST_PATH = os.path.join(INPUT, 'test')
#TEST_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/test"
CSV_PATH = os.path.join( INPUT, 'csvs' )
#CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/csvs"
DAT_PATH = os.path.join(INPUT, 'dats')
#DAT_FILE_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/dats"
TRAIN_CSV_PATH = os.path.join(CSV_PATH, 'train.csv')
#TRAIN_CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/csvs/train.csv"
TEST_CSV_PATH = os.path.join(CSV_PATH, 'test.csv')
#TEST_CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/csvs/test.csv"
TRAIN_FACIAL_LANDMARKS_CSV_PATH = os.path.join( CSV_PATH, 'train_landmarks.csv' )
#TRAIN_FACIAL_LANDMARKS_CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/csvs/train_landmarks.csv"
TEST_FACIAL_LANDMARKS_CSV_PATH = os.path.join( CSV_PATH, 'test_landmarks.csv')
#TEST_FACIAL_LANDMARKS_CSV_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/csvs/test_landmarks.csv"
SHAPE_PREDICTOR_DAT_PATH = os.path.join( DAT_PATH, 'shape_predictor_68_face_landmarks.dat' )
#SHAPE_PREDICTOR_DAT_PATH = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/input/dats/shape_predictor_68_face_landmarks.dat"
SAVE_MODEL = os.path.join( ROOT_DIR, 'models' )
#SAVE_MODEL = "M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/models"

BATCH_SIZE = 64
TARGET_SIZE = {
    'irnv2' : ( 224, 224, 3 ),
    'xception' : ( 224, 224, 3 ),
    'resnet50' : ( 197, 197, 3 ),
    'CV2_LANDMARKS_RESIZE' : ( 200, 200, 3 ),
    'customFacialLandmarks' : ( 68*2 )
}
#MODELS = [ 'irnv2', 'xception', 'resnet50' ]
MODELS = ['customFacialLandmarks']
NON_IMAGE_MODELS = ['customFacialLandmarks']
NUM_SPLITS = 5
LIST_OF_FOLD_EXCEPTIONS = [ idx + 1 for idx in range( 0, len( MODELS ) ) if MODELS[idx] in NON_IMAGE_MODELS]

CLASS_LABEL = {'Angry': 0, 'Disgust': 1, 'Fear': 2,
               'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
NUM_CLASSES = len( CLASS_LABEL )


LAYERS_TO_TRAIN = {
    'irnv2' : 774,
    'xception' : 126,
    'resnet50' : None
}

preprocessing_function = {
    'irnv2' : preprocess.irnv2,
    'xception' : preprocess.xception,
    'resnet50' : preprocess.resnet50,
    'custom_facialLandmarks' : preprocess.customFacialLandmarks
}
