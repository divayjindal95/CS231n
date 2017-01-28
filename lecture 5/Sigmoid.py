def sigmoid(z):
	return 1.0/(1+np.exp(-1.0*z))

def sigmoid_derivative(z,a):
	return a*(1-a)
