from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
from multiDim.Function_Rotation3D import Function_Rotation3D 
from multiDim.Function_Periodic_Behaviour import Function_Periodic_Behaviour
import torch
from multiDim.Approximator_Fourier_ND import Approximator_Fourier_ND
from multiDim.ApproximatorTransformer import Approximator_Transformer
import numpy as np
import torch.nn as nn
from multiDim.Function_Mandelbrot_2D import Function_Mandelbrot
from multiDim.Function_Basic1DArm import Function_Basic1DArm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available() :
    print("CUDA available")
else:
    print("CUDA NOT available")


def epochs_vs_loss(function,name,epochs_list=np.arange(1,300,20),nodesPerLayer=[8,8,8],samplePoints=10,logscale=False):
    loss_vs_epochs = Experiment_ND(name,[],function,logscale=logscale)
    loss_vs_epochs.plot_norms_vs_epochs(epochs_list,samplePoints,nodesPerLayer)

def time_vs_epochs_n_samplePoints(function,name="epochen_samplepoints_map",sample_points_range=[1,1000],epochs_range=[1,1000],nodes_per_layer=[4,4],n_random_samples=50,activation_function=torch.nn.ReLU(),loss_fn_class=torch.nn.MSELoss):
    Experiment_ND("test",[],function).plot_training_time_heatmap_random_sampling(name,function,activation_function,loss_fn_class,epochs_range, sample_points_range,nodes_per_layer,n_random_samples)

def startCasualExp():
    exp = Experiment_ND("Test",getApprox(),getFunc(),parallel=False,logscale=False,loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.print_loss_summary(mode="mse")
    exp.print_loss_summary(mode="l1")
    exp.print_loss_summary(mode="max")
    exp.plot_error_histograms()
    exp.plot_pca_querschnitt_all_outputs()
    exp.plot_1d_slices(mode="median")

def getApprox():
    approx_shepard = ShepardInterpolator([],300,power=3)
    approx_identity = Approximator_Identity_ND([])
    #approx_transformer = Approximator_Transformer( params=[500, 500, [16, 16]], device = device)
    apprx = [approx_identity,approx_shepard]
    #return apprx
    
    for i in {3,10,300}:
        for j in {100000}:
            for k in {1e-1,1e-2}:
                apprx.append(Approximator_Fourier_ND(params=[j,i],ridge_lambda=k))
    #return apprx
    #apprx.append(Approximator_Transformer(params=[800,10000,[4,4]],num_layers=2,name ="Transformer"))

    #return apprx
    for i in {9000}:
        for j in {1000}:
            apprx.append(Approximator_NN_ND([i,j,[16,16]],activationFunction = nn.Sigmoid()))
    return apprx

def getFunc():
    function_rotation = Function_Rotation3D()
    function_multiDim=Function_MultiDimOutput()
    function_periodic = Function_Periodic_Behaviour()
    function_sin_2D = Function_Sin_2D()
    function_BasicArm= Function_Basic1DArm()
    function_sin_4D = Function_Sin_4D()
    function_mandel = Function_Mandelbrot()
    return function_BasicArm

#exp = Experiment_ND("Fourier_Frequenzen_vs_Loss",[],getFunc(),logscale=True)
#exp.plot_norms_vs_fourier_freq(how_many_points_on_plot= 15,parallel=False,max_freqs=300,ridge_rate=1e-1,samplePoints=20000)
#startCasualExp()
def exp_robo_function():
    print("Robo")
    exp = Experiment_ND("Robo",getApprox(),getFunc(),parallel=False,logscale=False,loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.visualize_6D_poses_in_3D()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()

def exp_rotation_3D_function():
    print("Rotation")
    exp = Experiment_ND("Rotation_3D",getApprox(),Function_Rotation3D(),loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()
    exp.plot_vector_fields_3D_all()


def exp_sinus_2D_function():
    print("Sinus_2D")
    exp = Experiment_ND("Sinus_2D",getApprox(),Function_Sin_2D(),loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()
    exp.visualize2D()
    
def exp_sinus_4D_function():
    print("Sinus_4D")
    exp = Experiment_ND("Sinus_4D",getApprox(),Function_Sin_2D(),loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()    

def exp_plotting_loss_vs_epochs():
    print("NN_epochs")
    exp = Experiment_ND("NN_epoch_vgl",[],Function_Periodic_Behaviour(),logscale=True,parallel=True)

    exp.plot_norms_vs_epochs([1,10,100,200,400,800,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,10000],10000,[16,16,16])

def exp_plotting_loss_vs_frequencies():
    print("Fourier_Frequ")
    exp = Experiment_ND("Fourier vgl",[],Function_Periodic_Behaviour(),logscale=True)

    exp.plot_norms_vs_fourier_freq(ridge_rate=0.1,max_freqs=300,how_many_points_on_plot=20,parallel=False)

#exp_plotting_loss_vs_frequencies()
#exp_plotting_loss_vs_epochs()
#exp_sinus_2D_function()
#exp_sinus_4D_function()
#exp_robo_function()
#exp_rotation_3D_function()
