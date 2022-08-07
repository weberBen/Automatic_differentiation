
import numpy as np
#%%

from .Vector import Vector

#%%

def computeJacobianBackward(inputs, func_res):
    """
    Compute the jacobian of function f(a1, a2, a3, ..., an) = (f1(a1, ..., an), ...., fm(a1, ..., an)) from R^n to R^m in backward mode

    Parameters
        ----------
            inputs : array
                list of Vector inputs (a1, a2, a3, ..., an)
            func_res : tuples
                list of function outputs (f1(a1, ..., an), ...., fm(a1, ..., an))
    
    Return
    ----------
        jacobian : matrix
            the jacobian of the function R^n to R^m
    """

    number_inputs = len(inputs)
    number_outputs = len(func_res)

    jacobian = np.zeros((number_inputs, number_outputs))

    # Since the computation graph is shared by all the variables have been involved in operations, 
    # when computing function f(a1, a2, a3, ..., an) = (f1(a1, ..., an), ...., fm(a1, ..., an)) from R^n to R^m 
    # then the computation graph of f(a1, a2, a3, ..., an) contains all operations where a1,...,an are involved in f1,...fm
    # When requiring the computation graph of a specific output fi of f, then output node of fi becomes the only output and all the other
    # become unused nodes that will be removed. Thus, by computing the gradient of fi all operations made in f1,...fi-1,fi+1,...,fm
    # will be removed if they are not present in fi
    # So the resulting computation graph where the gradient is being computed will be only the one for fi as if there was the only
    # one expression
    # This is possible because cleaning the computation graph for computing the gradient makes an initial copy of the actual computation
    # graph. Then, after switching to another fj the underlaying computation graph with all the operations on the outputs remains the same

    # Todo: optimize the jacobian computation by doing it on all the output nodes at once instead of removing each
    # output except one to compute the gradient of a particular fi

    for index, fi in enumerate(func_res):
        fi.backward()

        for a_index, a in enumerate(inputs):
            jacobian[index][a_index] = a.grad
    
    return jacobian

#%%

def computeJacobianForward():
    # Not possible to implent it with dual number for the moment
    # since that for each output fi the same dual number is used
    raise NotImplementedError()