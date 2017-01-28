def numerical_gradient(x,f):
	fx=f(x)
	grad=np.zeros(x.shape)
	h=0.00001
	
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
   while not it.finished:
		
    # evaluate function at x+h
   	ix = it.multi_index
    	old_value = x[ix]
    	x[ix] = old_value + h # increment by h
    	fxh = f(x) # evalute f(x + h)
    	x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

	return grad
