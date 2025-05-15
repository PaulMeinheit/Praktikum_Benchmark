from package1.Sin2D_Function import Sin2D_Function 
from package1.Identity_Approximator import Identity_Approximator
from package1.Experiment import Experiment
from package1.Approx_Linear_Regression import Approx_Linear_Regression
from package1.Function_Chess import Function_Chess
from package1.Approx_NN import NN_Approximator
import numpy as np

sinFunc = Sin2D_Function("Sinus(x*y)", (-2*(np.pi)),(2*(np.pi)),(-2*(np.pi)),(2*(np.pi)))
function_chess = Function_Chess("Chess field",0,8,0,8)
approx2 = Approx_Linear_Regression("Linear Model",[1000])
approx3 = Identity_Approximator("Id3",[])
approxList = []
for i in range(7,8):
    approxList.append(NN_Approximator("Neuronales Netz mit Layers: ",[3000,100,[i,i,i]]))
exp = Experiment(approxList, sinFunc)
exp.run()