import numpy as np
import pandas as pd
from scipy.sparse import construct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plot

def accuracy_of_model( y_true, y_pred ):
    print( "The accuracy of the model is: ", accuracy_score( y_true, y_pred ) )

def classificationReport_of_model( y_true, y_pred ):
    print( "-------------Classification Report--------------" )
    print( classification_report( y_true, y_pred ) )

def confusionMatrix_of_model( y_true, y_pred ):
    cf_matrix = confusion_matrix( y_true, y_pred )
    plot.plot_confusionMatrix( cf_matrix )

