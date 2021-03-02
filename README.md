# FER Codes and Datasets

The scripts are portable enough to be shifted to any other system without any changes. 

The steps to be followed to emulate the codes are:

1. Run `prepare_data.py` file from `src`.
2. Run `train.py` file from `src`.
3. Test the model using `test.py`.

`config.py` ---> Used to define all the generics like train path, test path, model list, no of splits, class labels, no of layers to fine-tune fo reach model and return respective preprocessing functions.

`preprocess.py` ---> used to define various preprocessing functions for the models which are specified in `config.py`.

`model_from_weights.py` ---> Used to contruct the whole model if only pre-trained weights are available.

`model_dispatcher.py` ---> Used to define the model architecture and dispatch the models as called by `train.py`.

`train.py` ---> Used to train the models defined in `config.py` and dispatched by `model_dispatcher.py`.

`plot.py` ---> Used to define various eclectic plots concerned to the model performance.

`test.py` ---> Used for testing purpose.

`evaluation.py` ---> Used to define the evaluation criterias.


`config.py` file can be used to define which models are to be trained by specifying `MODELS = [..]` given the condition that the model is already implemented in the `model_dispatcher.py`

`StratifiedKFold` is used with `N_SPLITS = 5` so as to provide with `80-20` split of the dataset. The `N_SPLITS` can be changed from the `config.py` file.
