from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
import numpy as np


def getApprox():
    approx_nn_nd = Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[20000,1000,[64,64,64,64]])
    approx_nn_nd2 = Approximator_NN_ND("Versuch NN 64-64-64-64 & 7k Epochen & 400^n Punkten",[20000,10000,[64,64]])
    
    
    approx_shepard = ShepardInterpolator([],100000)
    approx_identity = Approximator_Identity_ND([])
    apprx = [approx_shepard,approx_nn_nd,approx_nn_nd2]
    return apprx

def getFunc():
    function_multiDim=Function_MultiDimOutput()
    function_sin_2D = Function_Sin_2D("Sinus-2D",2,1,[0,0],[2*np.pi,2*np.pi])
    function_sin_4D = Function_Sin_4D()

    return function_sin_4D 


exp = Experiment_ND("Test",[],getFunc())
exp.train()
epoch_list = [100,110,120,130,140,150,160,170,180,190,200]
exp.plot_norms_vs_epochs(epoch_list,1000,[32,32,32])
#exp.print_loss_summary()
#exp.plot_1d_slices()
#exp.plot_pca_querschnitt_all_outputs()