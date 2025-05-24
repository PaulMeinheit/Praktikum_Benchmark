from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
from multiDim.Function_Rotation3D import Function_Rotation3D 

import numpy as np
import torch


def getApprox():
    approx_nn_nd = Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[20000,1000,[64,64,64,64]])
    approx_nn_nd2 = Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[20000,10000,[64,64]])
    
    
    approx_shepard = ShepardInterpolator([],100000)
    approx_identity = Approximator_Identity_ND([])
    apprx = [approx_shepard,approx_nn_nd,approx_nn_nd2]
    return apprx

def getFunc():
    function_rotation = Function_Rotation3D()
    function_multiDim=Function_MultiDimOutput()
    function_sin_2D = Function_Sin_2D("Sinus-2D",2,1,[0,0],[2*np.pi,2*np.pi])
    function_sin_4D = Function_Sin_4D()

    return function_sin_4D 
epochs = np.arange(1,2000,100)
#loss_vs_epochs = Experiment_ND("Test",[],getFunc())
#loss_vs_epochs.plot_norms_vs_epochs(epochs,100,[4,4])

exp = Experiment_ND("Test",getApprox(),getFunc())
exp.train()
exp.print_loss_summary()
exp.plot_1d_slices()
exp.plot_pca_querschnitt_all_outputs()