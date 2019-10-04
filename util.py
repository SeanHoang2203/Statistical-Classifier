import numpy as np
from math import log,e,exp,pi

def to_categorical(x):
    """An utility function to convert to categorical form\n
    For example:
    >>> y = [1,2,3,4,3,2,3,1,4]
    >>> to_categorical(y)
    array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 1.]])
    """
    x = np.asarray(x)
    x_out = np.zeros((x.shape[0],len(set(x.tostring()))))
    for idx in range(x.shape[0]):
        x_out[idx,x[idx]] = 1
    return x_out


def max_count(a,y):
    """Return the most common element in an 1D numpy array """
    
    a_list = [y[a[idx],0] for idx in range(a.shape[0])]
    return max([(i,a_list.count(i)) for i in set(a_list)], key = lambda x: x[1])[0]

def print_result(train_hist,eval_hist,value_list = None, *args):
    for arg in args:
        idx =  value_list.index(arg)
        print('Training error and accuracy for {} is: '.format(arg),train_hist[idx],'\n',
        'Test error and accuracy for {} is: '.format(arg),eval_hist[idx])