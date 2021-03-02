# FER Codes and Datasets

The codes which are written are portable enough to be shifted to any other system without any changes. 

The steps to be followed to emulate the codes are:

1. Run the `prepare_data.py` file from `src`.
2. Run `train.py` file from `src`.
3. Test the model using `test.py`.


`config.py` file can be used to define which model to be trained by specifying `MODELS = [..]` given the condition that the model is already implemented in the `model_dispatcher.py`
