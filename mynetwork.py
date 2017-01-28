import numpy as np
import pandas as pd
import random


def cost_derivative(a,Y):
	return a-Y
		
def mse(a,Y):
	return np.sum((a-Y)**2)

def sigmoid(z):
	return 1.0/(1+np.exp(-1.0*z))

def relu(z):
	return np.maximum(0,z)

def cross_entropy(a,Y):
	a=a-Y
	return np.exp(np.max(a))/np.sum(np.exp(a))

def preprocess(X):
	return (X-np.mean(X))/np.std(X)

class Network:
	def __init__(self,sizes,epoches,batch_size,eta):
		self.sizes=sizes
		self.num_layers=len(sizes)
		self.weights=[np.random.randn(y,x)/np.sqrt(x) 			#xavier initialization , use /2 with relu		
					for x,y in zip(self.sizes[:-1],self.sizes[1:])]
		self.biases=[np.random.randn(y,1)/np.sqrt(y) for y in self.sizes[1:] ]
		self.activations=[np.zeros(self.sizes) for size in self.sizes]
		self.epoches=epoches
		self.batch_size=batch_size
		self.eta=eta


	def backprop(self,activations,Z,X,Y):
		weight_gradient=[np.zeros((y,x)) 
						for x,y in zip(self.sizes[:-1],self.sizes[1:])]
		bias_gradient=[np.zeros((y,1)) for y in self.sizes[1:] ]

		# Use your cost function here
		cost=mse(activations[-1], Y) 
		gradient=cost_derivative(activations[-1],Y)*sigmoid(Z[-1])*(1-sigmoid(Z[-1]))

		weight_gradient[-1]=np.dot(gradient,activations[-2].T)
		bias_gradient[-1]=gradient
		for l in xrange(2, self.num_layers):
			gradient=np.dot(self.weights[-l+1].T,gradient)
			weight_gradient[-l]=np.dot(gradient,activations[-l-1].T)
			bias_gradient[-l]=gradient
		return bias_gradient , weight_gradient

	def feed_forward(self, X):
		activations = [X]
		Z=[X]
		for l in xrange(self.num_layers-1):
			z = np.dot(self.weights[l], activations[l])+self.biases[l]

			#Use your Activation function here
			a = sigmoid(z)

			activations.append(a)
			Z.append(z)
		return activations,Z
	
	def update_mini_batch(self,mini_batch):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			Y=np.zeros((10,1))
			X=np.reshape(x,(len(x),1))
			X=preprocess(X)
			Y[y-1]=1
			activations,Z=self.feed_forward(X)
			delta_nabla_b, delta_nabla_w = self.backprop(activations,Z,X,Y)
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(self.eta/len(mini_batch))*nw
						for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(self.eta/len(mini_batch))*nb
						for b, nb in zip(self.biases, nabla_b)]
	
	def SGD(self,training_data):
		n=len(training_data)
		for j in xrange(self.epoches):
			print j
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+self.batch_size] 
							for k in xrange(0, n, self.batch_size)]
			for mini_batch in mini_batches:
				self.update_mini_batch(mini_batch)

if __name__=="__main__":

	import cPickle
	f=open('./cifar-10-batches-py/data_batch_1')   
	dict1=cPickle.load(f)
	training_data=zip(dict1['data'],dict1['labels'])

	net=Network([3072,100,10],batch_size=20,epoches=3,eta=0.001)
	net.SGD(training_data=training_data)
