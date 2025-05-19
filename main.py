from package1.Sin2D_Function import Sin2D_Function 
from package1.Identity_Approximator import Identity_Approximator
from package1.Experiment import Experiment
from package1.Approx_Linear_Regression import Approx_Linear_Regression
from package1.Function_Chess import Function_Chess
from package1.Function_Rosenbrock import Function_Rosenbrock
from package1.Approx_NN import NN_Approximator
from package1.Function_Ackley import Function_Ackley
from package1.Function_Heart import Function_Heart
import numpy as np
function_heart = Function_Heart("Herz-Shape",-1.5,1.5,-1.5,1.3)

function_ackley = Function_Ackley("Ackley Function",-4,4,-4,4)
function_chess = Function_Chess("Chess field",0,8,0,8)
function_sinChess = Sin2D_Function("sin(x)*cos(y)", (-2*(np.pi)),(2*(np.pi)),(-2*(np.pi)),(2*(np.pi)))
function_rosenbrock = Function_Rosenbrock("Rosenbrock",-2,2,-1,3)
approx_lin_1000 = Approx_Linear_Regression(f"Linear Model (1000 Samples)",[1000])
approx_lin_500 = Approx_Linear_Regression(f"Linear Model (500 Samples)",[500])

approx_id = Identity_Approximator("Identity",[])

approx_nn_E500_L8_8_8 = NN_Approximator("8er",[1200,50,[8,8,8,8]])
approx_nn_E500_L4_4_4 = NN_Approximator("4er",[1200,50,[4,4,4]])
approx_nn_E500_L16_16_16 = NN_Approximator("16er",[1200,50,[16,16,16]])

#Rosenbrock experiment:
exp_rosenBrock_list = []
approxList = [approx_id]
for j in {8,16,32}:
    name = f"NN-Epochs: {100} -L: {j},{j},{j} (fully connected)"
    approxList.append(NN_Approximator(name,[100,150,[j,j,j]]))

for i in range(1,4):
    exp_rosenBrock_list.append(Experiment(f"Rosenbrock-Funktion mit {i*100} Epochen",True,True, approxList, function_rosenbrock))
for exp in exp_rosenBrock_list:
    exp.run()