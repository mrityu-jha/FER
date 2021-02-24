import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss( history ):
    plt.figure( figsize = ( 8, 7 ) )
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss', fontsize = 14 )
    plt.ylabel('Loss', fontsize = 18 )
    plt.xlabel('Epoch', fontsize = 17 )
    plt.xticks( fontsize = 14 )
    plt.yticks( fontsize = 14 )
    plt.legend( ['Train Set', 'Val Set'], loc='best', fontsize = 14 )
    plt.show()



def plot_accuracy( history ):
    plt.figure( figsize = ( 8, 7 ) )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy', fontsize = 14 )
    plt.ylabel('Accuracy', fontsize = 18 )
    plt.xlabel('Epoch', fontsize = 17 )
    plt.xticks( fontsize = 14 )
    plt.yticks(fontsize = 14 )
    plt.legend( ['Train Set', 'Val Set'], loc='best', fontsize = 14 )
    plt.show()
