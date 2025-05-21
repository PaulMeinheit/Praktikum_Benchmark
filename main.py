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

def getApproxs():
    approx_lin_1000 = Approx_Linear_Regression(f"Linear Model (1000 Samples)",[1000])
    approx_lin_500 = Approx_Linear_Regression(f"Linear Model (500 Samples)",[500])

    approx_id = Identity_Approximator("Identity-Approximation",[])

    approx_nn_E500_L8_8_8 = NN_Approximator("8er",[1200,50,[8,8,8,8]])
    approx_nn_E500_L4_4_4 = NN_Approximator("4er",[1200,50,[4,4,4]])
    approx_nn_E500_L16_16_16 = NN_Approximator("16er",[1200,50,[16,16,16]])

    approxList = [approx_id]
    for j in {16,32}:
        for i in {100,500,6000}:
            name = f"NN-Aufbau: {j}-{j}-{j} (fully connected)\n& Epochen: {i}"
            approxList.append(NN_Approximator(name,[i,150,[j,j,j]]))
    
    return approxList


#Alle Experimente laufen lassen:
exp_list = []

exp_list.append(Experiment(f"Rosenbrock-Funktion",True,True, getApproxs(), function_rosenbrock,vmin=1e-5,vmax=1e4))

exp_list.append(Experiment(f"Ackley-Funktion",True,False, getApproxs(), function_ackley))

exp_list.append(Experiment(f"Schach-Funktion",True,False, getApproxs(), function_chess))

exp_list.append(Experiment(f"Sinus-Schach-Funktion",True,False, getApproxs(), function_sinChess))

exp_list.append(Experiment(f"Herz-Funktion",True,False, getApproxs(), function_heart))

for exp in exp_list:
    exp.run()