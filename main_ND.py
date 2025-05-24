from multiDim.Experiment_ND import Experiment_ND
from multiDim.Approximator_NN_ND import Approximator_NN_ND
from multiDim.Function_SinChess_4D import Function_Sin_4D
from multiDim.Function_SinChess_2D import Function_Sin_2D
from multiDim.Function_multiDimOutput import Function_MultiDimOutput
import numpy as np


exp = Experiment_ND("Test-Experiment",[Approximator_NN_ND("Versuch NN 4-4-1",[7000,400,[64,64,64,64]])],Function_MultiDimOutput())
exp.train()
exp.print_loss_summary()
exp.plot_1d_slices()
exp.plot_pca_querschnitt_all_outputs()
