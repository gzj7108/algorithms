"""
optimization methods

"""

import numpy

def gradient_descent(o_x,g,l_r):
    """gradient descent optimization
    input  old parameter,gradient,learning_rate
    output new parameter
    date:2020/1/17
    """
    
    return o_x-g*l_r