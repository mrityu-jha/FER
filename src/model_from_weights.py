import tensorflow as tf
import model_dispatcher
import config
import os 

def create_model( path, name_of_model ):
    model = model_dispatcher.return_model( None, name_of_model )
    print( model.summary() )    
    print( 'Loading Weights from: ', path )
    model.load_weights( path )
    model.compile()
    print( "Weights Loaded into the model" )
    tf.saved_model.save(
        model,
        os.path.join( config.SAVE_MODEL, 'best_' + name_of_model + '_preTrained' ) 
    )


if __name__ == '__main__':
    path = 'M:/My Files/Mrityunjay Jha/Programming/MIBM Lab/Facial Expression Recognition/ResNet-50.h5'
    name_of_model = 'resnet50'
    create_model( path, name_of_model )

