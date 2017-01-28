#Mean squared error
import numpy as np
class MSE():
	
	def cost(self,a,y):
		return np.sum((a-y)**2)

	def cost_derivative(self,a,y):
		return a-y
			

