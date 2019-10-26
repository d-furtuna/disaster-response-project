'''
    Custom scoring function used in GridSearchCV.
    Function is the percentual sum of true negatives and true positives.
    Custom_scoring_function function is saved in a user-created module to use n_jobs greater than 1.
'''

from sklearn.metrics import multilabel_confusion_matrix
import numpy as np

def custom_scoring_function(Y_test, Y_pred):
    confusion_mat = multilabel_confusion_matrix(Y_test, Y_pred)
    total_accuracy = np.trace(confusion_mat, axis1=1, axis2=2).sum() / confusion_mat.sum()
    
    return total_accuracy