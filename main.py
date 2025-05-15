from package1.Sin2D_Function import Sin2D_Function 
from package1.Identity_Approximator import Identity_Approximator
from package1.Experiment import Experiment
from package1.Approx_Linear_Regression import Approx_Linear_Regression
from package1.Function_Chess import Function_Chess
from package1.Function_Rosenbrock import Function_Rosenbrock
from package1.Approx_NN import NN_Approximator
import numpy as np

function_chess = Function_Chess("Chess field",0,8,0,8)
function_sinChess = Sin2D_Function("sin(x)*cos(y)", (-2*(np.pi)),(2*(np.pi)),(-2*(np.pi)),(2*(np.pi)))
function_Rosenbrock = Function_Rosenbrock("Rosenbrock",-2,2,-1,3)
approx_lin_1000 = Approx_Linear_Regression(f"Linear Model (1000 Samples)",[1000])
approx_lin_500 = Approx_Linear_Regression(f"Linear Model (500 Samples)",[500])

approx_id = Identity_Approximator("Identity",[])

approx_nn_E500_L4_4_4 =NN_Approximator("name",[500,100,[4,4,4]])
#sin-function experiment:
approxList = []
for i in [4,8,12,16,32,64,128]:
    for j in [1,2,4,8,16,32]:
        name = f"Neuronales Netz E: {100*j} & L: {i} (fully connected)"
        approxList.append(NN_Approximator(name,[10*j,100,[i,i,i]]))
exp = Experiment(True, approxList, function_Rosenbrock)
exp.run()