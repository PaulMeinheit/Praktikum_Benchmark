from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
import numpy as np


def getApprox():
    approx_nn_nd =Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[40000,1000,[64,64,64,64]])
    approx_shepard = ShepardInterpolator([],100000)
    approx_identity = Approximator_Identity_ND([])
    apprx = [approx_identity,approx_shepard, approx_nn_nd]
    return apprx

def getFunc():
    function_multiDim=Function_MultiDimOutput()
    return function_multiDim


exp = Experiment_ND("Test",getApprox(),getFunc())
exp.train()
exp.print_loss_summary()
exp.plot_1d_slices()
exp.plot_pca_querschnitt_all_outputs()