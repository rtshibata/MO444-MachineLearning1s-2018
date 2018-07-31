# Modules
import numpy as np
import pandas as pd
import sys

class NormalEquations:

	#construtor
	def __init__(self,X,target_values,lambda_):
		self.data = X
		self.lambda_ = lambda_
		self.target_values = target_values
		self.theta = pd.DataFrame() #empty
		
	def begin_alg(self):
		y = self.target_values
		m,n = self.data.shape

		# Normal Equation with regularization:
		# theta = inv(X^T * X + lambda_I) * X^T * y
		theta = []

		# 		I = |0 ....  0|  --> theta0 is not regularized
		#		    |. 1 ... 0|
		#			|. . 1 ...|
		#			|.  ...  0| 
		#			|0 ... 0 1|

		#n and not (n+1) for theta0 is already being counted
		identity_neq = np.identity(n) 
		identity_neq[0,0] = 0
		identity_neq = self.lambda_ * identity_neq
		lambda_I = pd.DataFrame(data=identity_neq)

		#print(lambda_I)

		# Calculating theta
		X_transpose = self.data.transpose()
		#print(X_transpose.shape)
		XtX = X_transpose.dot(self.data)
		#print(XtX.shape)
		theta = XtX + lambda_I
		#print(theta.shape)
		try:
			theta = np.linalg.inv(theta)
			#print(theta.shape)
			theta = theta.dot(X_transpose)
			#print(theta.shape)
			theta = theta.dot(y)
			self.theta = theta
		except np.linalg.LinAlgError as err:
			print("Matriz XtX nao eh invertivel, nao eh possivel calcular theta por Normal equations")
			sys.exit(0)

	def get_theta(self):
		#print self.theta
		return self.theta

	'''
	def begin_test(self):
		X = self.data.iloc[:,:-1]
		m,n = X.shape
		#mean squared error
		for i in range(n):
			X_i = X.iloc[i,:]
			y_f = theta.dot(X_i)
			mse = y_f - y_test.iloc[i]
			mse = float(mse ** 2)/float(n)

		print mse
	'''





