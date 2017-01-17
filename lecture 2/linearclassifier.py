import cPickle
import numpy as np
import cv2
from sklearn.svm import LinearSVC


f=open('/home/divay/Documents/cs321n/cifar-10-batches-py/data_batch_1')
pkl=cPickle(f)
dict1=pkl.load()
x=dict1['data']
y=dict1['label']


clf=LinearSVC()
clf.fit(x,y)
w=clf.coeff_

w=w*1000000*255/172
cat=w[0]
rs_w=np.reshape(cat,(32,32,3)
rs_w=cv2.resize(rs_w,(320,320))
cv2.imshow('',rs_w)
