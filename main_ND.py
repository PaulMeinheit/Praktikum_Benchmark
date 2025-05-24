from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
import numpy as np


def getApprox():
    approx_nn_nd =Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[1000,100,[64,64,64,64]])
    shepard = ShepardInterpolator([],100)
    apprx = [shepard, approx_nn_nd]
    return apprx
exp = Experiment_ND("Test-Experiment",getApprox(),Function_MultiDimOutput())
exp.train()
exp.print_loss_summary()
exp.plot_1d_slices()
exp.plot_pca_querschnitt_all_outputs()
