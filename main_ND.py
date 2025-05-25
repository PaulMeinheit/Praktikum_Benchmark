from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
from multiDim.Function_Rotation3D import Function_Rotation3D 
import torch
import numpy as np


def epochs_vs_loss(epochs_list,name,function,nodesPerLayer=[8,8,8],samplePoints=10,logscale=False):
    loss_vs_epochs = Experiment_ND(name,[],function,logscale=logscale)
    loss_vs_epochs.plot_norms_vs_epochs(epochs_list,samplePoints,nodesPerLayer)

def time_vs_epochs_n_samplePoints(function,name="epochen_samplepoints_map",sample_points_range=[1,1000],epochs_range=[1,1000],nodes_per_layer=[8,8,8],n_random_samples=100,mse_threshold=1e-1,activation_function=torch.nn.ReLU(),loss_fn_class=torch.nn.MSELoss):
    Experiment_ND("test",[],function).plot_training_time_heatmap_random_sampling(name,function,activation_function,loss_fn_class,epochs_range, sample_points_range,nodes_per_layer,n_random_samples,mse_threshold)





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
    return function_multiDim

epochs = np.arange(1,300,20)
#epochs_vs_loss(epochs,"random_test",getFunc())
time_vs_epochs_n_samplePoints(getFunc())
#exp = Experiment_ND("Test",getApprox(),getFunc(),parallel=False,logscale=False)


#exp.train()
#exp.print_loss_summary()
#exp.plot_1d_slices()
#exp.plot_pca_querschnitt_all_outputs()