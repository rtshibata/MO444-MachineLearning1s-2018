# Modules
import numpy as np
import pandas as pd
import sys

class NormalEquations:

	#construtor
	def __init__(self,X,target_values):
		self.data = X

		self.target_values = target_values
		self.theta = pd.DataFrame() #empty
		
	def begin_alg(self):

		y = self.target_values

		X = self.data
		m,n = X.shape

		theta = []
		#print X
	
		#print y
		# Normal Equation:
		# theta = inv(X^T * X) * X^T * y

		# For convenience I create a new, tranposed X matrix
		X_transpose = X.transpose()

		# Calculating theta
		XtX = X_transpose.dot(X)
		try:
			theta = np.linalg.inv(XtX)
			theta = theta.dot(X_transpose)
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





