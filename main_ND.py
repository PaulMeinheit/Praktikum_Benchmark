from tracemalloc import start
from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
from multiDim.ShepardInterpolator import ShepardInterpolator
from multiDim.Approximator_Identity_ND import Approximator_Identity_ND
from multiDim.Function_Rotation3D import Function_Rotation3D 
from multiDim.Function_Periodic_Behaviour import Function_Periodic_Behaviour
from multiDim.Function_Exponential import Function_Exponential
from multiDim.Function_Lin import Function_Lin
from multiDim.Function_Polynom import Function_Polynom



import torch
from multiDim.Approximator_Fourier_ND import Approximator_Fourier_ND
from multiDim.ApproximatorTransformer import Approximator_Transformer
import numpy as np
import torch.nn as nn
from multiDim.DGL_Visualizer import DGL_Visualizer
from multiDim.Function_Mandelbrot_2D import Function_Mandelbrot
from multiDim.Function_Basic1DArm import Function_Basic1DArm
from multiDim.Function_DGL import Function_Lorentz_DGL
import time

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
    exp = Experiment_ND("Test",getApprox(),getFunc(),parallel=True,logscale=True,loss_fn=torch.nn.SmoothL1Loss(),max_workers=4)
    exp.train()
#    exp.print_loss_summary(mode="mse")
#    exp.print_loss_summary(mode="l1")
#    exp.print_loss_summary(mode="max")
    exp.plot_error_histograms()
    #exp.plot_pca_querschnitt_all_outputs()
    exp.plot_1d_slices(mode="median")

def getApprox():
    apprx = []
    #approx_transformer = Approximator_Transformer( params=[500, 500, [16, 16]], device = device)
    for i in {5000,10000,30000}:
        for j in {4,5}:
            apprx.append(ShepardInterpolator([],i,power=j))
    
    apprx.append(Approximator_NN_ND([18000,1500,[16,16,16,16]]))
    apprx.append(Approximator_NN_ND([10000,4000,[32,32]]))
    for i in {30000}:
        for j in {5000}:

            apprx.append(Approximator_NN_ND([i,j,[2,2]]))
            apprx.append(Approximator_NN_ND([i,j,[4,4]]))
            apprx.append(Approximator_NN_ND([i,j,[8,8]]))
            apprx.append(Approximator_NN_ND([i,j,[8,8]]))
            apprx.append(Approximator_NN_ND([i,j,[16,16]]))
            apprx.append(Approximator_NN_ND([i,j,[32,32]]))
            apprx.append(Approximator_NN_ND([i,j,[64,64]]))
            apprx.append(Approximator_NN_ND([i,j,[128,128]]))
            apprx.append(Approximator_NN_ND([i,j,[256,256]]))
            apprx.append(Approximator_NN_ND([i,j,[400,400]]))
    #Beste Approximatoren von Test mit Epochen,sample points
    apprx.append(Approximator_NN_ND([18000,1500,[16,16,16,16]]))
    apprx.append(Approximator_NN_ND([10000,4000,[32,32]]))
    apprx.append(Approximator_NN_ND([20000,6000,[32,32]]))
    apprx.append(Approximator_NN_ND([20000,6000,[64,64]]))
    
    return apprx
    for i in {300,3000}:
        for j in {10000,30000}:
           apprx.append(Approximator_Fourier_ND(params=[j,i],ridge_lambda=1e-2))
    return apprx


    apprx.append(Approximator_Transformer(params=[800,10000,[4,4]],num_layers=2,name ="Transformer"))
    return apprx

def getFunc():
    function_rotation = Function_Rotation3D()
    function_multiDim=Function_MultiDimOutput()
    function_periodic = Function_Periodic_Behaviour()
    function_sin_2D = Function_Sin_2D()
    function_BasicArm= Function_Basic1DArm()
    function_sin_4D = Function_Sin_4D()
    function_Lorentz_DGL = Function_Lorentz_DGL()
    function_mandel = Function_Mandelbrot()
    function_linear = Function_Lin()
    function_Polynom = Function_Polynom()
    function_exponential = Function_Exponential()

    return function_Lorentz_DGL

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
    exp = Experiment_ND("Sinus_4D",getApprox(),Function_Sin_4D(),loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()    

def exp_dgl_function():
    print("Lorentz_Attraktor")
    exp = Experiment_ND("Lorentz_Attraktor",getApprox(),Function_Lorentz_DGL(),loss_fn=torch.nn.SmoothL1Loss())
    exp.train()
    exp.plot_error_histograms()
    exp.plot_1d_slices()
    exp.plot_pca_querschnitt_all_outputs()    

def exp_plotting_loss_vs_epochs():
    print("NN_epochs")
    exp = Experiment_ND("NN_epoch_vgl",[],Function_Lorentz_DGL(),logscale=True,parallel=False)

    exp.plot_norms_vs_epochs([1,10,100,200,400,800,1000,1200,1400,1600,1800,2000,2500,3000,3500,4000,10000,15000,20000],1500,[16,16,16])

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


def plotEpochsAndStuffVsFunction(function):
    experiment = Experiment_ND("Compare_Complexity",[],function,logscale=True,parallel=True)
    model_configs = [
        {
            "nodes_per_layer": [2, 2],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.MSELoss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [4,4],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [2, 2],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.MSELoss(),
            "lr": 0.001
        },
        {
            "nodes_per_layer": [4,4],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.001
        },
        {
            "nodes_per_layer": [8,8],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.001
        },
        {
            "nodes_per_layer": [16,16],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.001
        },
        {
            "nodes_per_layer": [4,4,4],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [32,32],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [32,32,32],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [128,128],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [256,256],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [412,412],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.005
        },
        {
            "nodes_per_layer": [64,64],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.001
        },
        {
            "nodes_per_layer": [64,64,64],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.01
        },
        {
            "nodes_per_layer": [128,128,128],
            "activation_function": torch.nn.ReLU(),
            "loss_fn": torch.nn.L1Loss(),
            "lr": 0.03
        }
    ]

    start = time.time()
    experiment.plot_error_vs_epochs(
    model_configs=model_configs,
    epoch_counts=[400,500,1000,2000,3000,4000,6000,8000,9000,10000,12000,14000,16000,18000,20000,25000,30000,35000,40000,50000,60000],
    fixed_samples=3000,parallel=True)

    print(f"Epochs-Time: {time.time() - start:.2f}s")
    #start = time.time()
    #experiment.plot_error_vs_samples(
    #    model_configs=model_configs,
    #    sample_counts=[1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,12000,14000,16000,18000,20000],
    #    fixed_epochs=10000,parallel=True
    #)

    #print(f"Samples-Time: {time.time() - start:.2f}s")

def clusterShit():
    for func in [Function_Sin_2D(), Function_Periodic_Behaviour(), Function_Sin_4D(), Function_Rotation3D(), Function_Mandelbrot(), Function_Basic1DArm(), Function_MultiDimOutput()]:
        plotEpochsAndStuffVsFunction(func)

def dgl_visualizer():
    dgl_visualizer = DGL_Visualizer("3D_Vector_Fields", getApprox(),Function_Lorentz_DGL(),loss_fn=torch.nn.SmoothL1Loss(), parallel= False)
    dgl_visualizer.train()

    dgl_visualizer.plot_trajectories_video(n_steps=1000,delta=0.004,fps=20,combine=False)

    dgl_visualizer.plot_trajectories_video(n_steps=1000,delta=0.004,fps=20,combine=False)
    dgl_visualizer.plot_trajectories_3D_all()
    exp_dgl_function()



def all_functions_plotting():
    plotEpochsAndStuffVsFunction(Function_Exponential())
    plotEpochsAndStuffVsFunction(Function_Lin())
    plotEpochsAndStuffVsFunction(Function_Polynom())
    plotEpochsAndStuffVsFunction(Function_Lorentz_DGL())
    plotEpochsAndStuffVsFunction(Function_Sin_4D())
    


#plotEpochsAndStuffVsFunction(Function_Lorentz_DGL())
#exp_plotting_loss_vs_epochs()
#dgl_visualizer()
#all_functions_plotting()
dgl_visualizer()
#startCasualExp()
#exp_sinus_4D_function()
