def Relu(a):
	np.maximum(0,a)

def relu_derivative(z,a):
	arr=[1.0]*len(a)
	arr=np.array(arr)
	arr=np.reshape(arr,(len(a),1))
	return arr*(Relu(a)>0)

def LeakyRelu(alpha,a):
	np.maximum(alpha*a,a)

def LeakyRelu_derivative(z,a,alpha):
	arr=[1.0]*len(a)
	arr=np.array(arr)
	arr=np.reshape(arr,(len(a),1))
	return arr*(LeakyRelu(alpha,a)>) #make changes
	
	
#def ParaRelu():

#def elu():
