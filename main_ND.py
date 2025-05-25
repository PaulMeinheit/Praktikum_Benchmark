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
    approx_nn_nd = Approximator_NN_ND("NN 4-4-4 E:2k",[9000,10000,[16,16,16]])
    approx_nn_nd2 = Approximator_NN_ND("NN 16-16 E:6k",[1000,10000,[16,16]])
    
    
    approx_shepard = ShepardInterpolator([],100000)
    approx_identity = Approximator_Identity_ND([])
    apprx = [approx_identity,approx_shepard,approx_nn_nd]
    return apprx

def getFunc():
    function_rotation = Function_Rotation3D()
    function_multiDim=Function_MultiDimOutput()
    function_sin_2D = Function_Sin_2D()
    function_sin_4D = Function_Sin_4D()
    return function_sin_4D

epochs = np.arange(1,500,30)
loss_vs_epochs = Experiment_ND("Test",[],getFunc(),logscale=False)
loss_vs_epochs.plot_norms_vs_epochs(epochs,300,[4,4,4])

#exp = Experiment_ND("Test",getApprox(),getFunc(),parallel=False,logscale=False)


#exp.train()
#exp.print_loss_summary()
#exp.plot_1d_slices()
#exp.plot_pca_querschnitt_all_outputs()