from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
from multiDim.Function_Rotation3D import Function_Rotation3D 
import torch
from multiDim.Approximator_Fourier_ND import Approximator_Fourier_ND
from package1.Transformer import Approximator_Transformer
import numpy as np


def epochs_vs_loss(function,name,epochs_list=np.arange(1,300,20),nodesPerLayer=[8,8,8],samplePoints=10,logscale=False):
    loss_vs_epochs = Experiment_ND(name,[],function,logscale=logscale)
    loss_vs_epochs.plot_norms_vs_epochs(epochs_list,samplePoints,nodesPerLayer)

def time_vs_epochs_n_samplePoints(function,name="epochen_samplepoints_map",sample_points_range=[1,1000],epochs_range=[1,1000],nodes_per_layer=[4,4],n_random_samples=50,activation_function=torch.nn.ReLU(),loss_fn_class=torch.nn.MSELoss):
    Experiment_ND("test",[],function).plot_training_time_heatmap_random_sampling(name,function,activation_function,loss_fn_class,epochs_range, sample_points_range,nodes_per_layer,n_random_samples)

def startCasualExp ():
    exp = Experiment_ND("Test",getApprox(),getFunc(),parallel=True,logscale=False,loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.print_loss_summary(mode="mse")
    exp.print_loss_summary(mode="l1")
    exp.print_loss_summary(mode="max")
    exp.plot_error_histograms()
    exp.plot_pca_querschnitt_all_outputs()
    exp.plot_1d_slices(mode="median")

def getApprox():
    approx_shepard = ShepardInterpolator([],50000,power=3)
    approx_identity = Approximator_Identity_ND([])
    approx_fourier = Approximator_Fourier_ND(params=[50000,100])
    approx_transformer = Approximator_Transformer(params=[500,500,[16,16,16]],num_layers=3,name ="Transformer")
    apprx = [approx_fourier]
    return apprx
    for i in {3000}:
        for j in {10000}:
            apprx.append(Approximator_NN_ND([i,j,[50,50],0.005]))

    return apprx

def getFunc():
    function_rotation = Function_Rotation3D()
    function_multiDim=Function_MultiDimOutput()
    function_sin_2D = Function_Sin_2D()
    function_sin_4D = Function_Sin_4D()
    return function_sin_4D

Experiment_ND("Fourier-Experiment",[],getFunc(),logscale=True,vmin=1e-15,vmax=1e50).plot_norms_vs_fourier_freq()
