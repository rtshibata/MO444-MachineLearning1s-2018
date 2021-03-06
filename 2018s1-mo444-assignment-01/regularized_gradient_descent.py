import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import normal_equations
import sys
import getopt

initial_path="~/Renato/Github"

train_csv = initial_path+ "/MO444-MachineLearning1s-2018/2018s1-mo444-assignment-01/data/train.csv"
test_csv = initial_path+"/MO444-MachineLearning1s-2018/2018s1-mo444-assignment-01/data/test.csv"
test_target = initial_path+ "/MO444-MachineLearning1s-2018/2018s1-mo444-assignment-01/data/test_target.csv"

#################################
#Usage:
#arguments
# -a = --alpha = alpha
# -l = --lambda = lambda
# -i = --iterations =  number of iterations
# -t = --threshold = threshold value of instances whose shares are above it are not considered for the training 
################################
def usage():
	print("---------------Usage--------------------")
	print("Pass the arguments -a -l -i -t\n(ou --alpha --lambda --iterations --threshold) followed by their respective POSITIVE NUMERIC value")

def is_positive_number(x):
	try:
		float(x)
	except ValueError:
		return False
	if float(x)>=0:
		return True
	else:
		return False

def get_parameters():	
	try:
		opts, args = getopt.getopt(sys.argv[1:],"a:l:i:t:",
			['alpha=', 'lambda=', 'iterations=', 'threshold='])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit()

	#argumentos vazios
	if len(opts)==0:
		usage()
		sys.exit()

	for opt, arg in opts:
		if opt in ('-a', '--alpha') and is_positive_number(arg):
			alpha= float(arg)
		elif opt in ('-l', '--lambda') and is_positive_number(arg):
			Lambda = float(arg)
		elif opt in ('-i','--iterations') and is_positive_number(arg):
			try:
				iterations = int(arg)
			except ValueError:
				print("iteration argument must be an integer!")
				sys.exit()
		elif opt in ('-t','--threshold') and is_positive_number(arg):
			threshold = float(arg)
		else:
			usage()
			sys.exit()

	'''
	print('alpha : {}'.format(alpha))
	print('lambda : {}'.format(Lambda))
	print('iterations : {}'.format(iterations))
	'''
	return alpha,Lambda,iterations,threshold

alpha, Lambda, iterations,threshold = get_parameters()


#Reads data
#data_matrix = pd.read_csv(path_2_csv_ic)
data_matrix = pd.read_csv(train_csv)

header_line = data_matrix.columns
#Drops columns URL and timedelta 
data_matrix.drop(data_matrix.columns[[0 , 1]], axis=1, inplace = True)
#Insert bias equals to 1
data_matrix.insert(0, "x0", 1)


#Column index of the target value
n=col_target = 59
#Number of instances
m=31715

numpy_data = data_matrix.as_matrix()
y = numpy_data[:,col_target]
y = y.reshape(m,1)
#Drops the target values column from training data
numpy_data = np.delete(numpy_data, [col_target], axis=1)

#Plot graph
print("Minimum share: {}  -- ----  Maximum share:{}".format(y.min(),y.max()))
print("Mean share: {}  -- ----  Standard Deviation share:{}".format(np.mean(y),np.std(y)))


k, bins, patches = plt.hist(y, bins = 100, range = (-1,y.max()))
plt.xlabel('Shares bin(target)', fontsize=15)
plt.ylabel('Number of URLs(instances) per bin', fontsize=15)

plt.show()
print("Percentage of instances presented in the above graph: " + repr( float(k.sum())/float(m)))

k, bins, patches = plt.hist(y, bins = 10, range = (-1,7000))
plt.xlabel('Shares bin(target)', fontsize=15)
plt.ylabel('Number of URLs(instances) per bin', fontsize=15)

plt.show()

#Use only representative instances, shares above a threshold value
threshold = 7000
outliers_index = np.argwhere(y > threshold)
outliers_index = np.delete(outliers_index, [1], axis=1)
m_selected_inst= len(y) - len(outliers_index)
print("Percentage of instances selected by share threshold {}: {} ".format(threshold,float(m_selected_inst)/float(m)))
summed_shares = y[y<=threshold].sum()

print("Total shares summed up by share threshold {}: summed up {}/total shares {}= {}".format(threshold,summed_shares,y.sum(),float(summed_shares)/float(y.sum())))

#Get rid of non-representative instances   
y = np.delete(y, outliers_index, axis = 0)
numpy_data = np.delete(numpy_data, outliers_index, axis=0)

train_data=numpy_data
############################
#Scaling: Mean Normalization
###########################
maximum_features = np.zeros([1, n])
minimum_features = np.zeros([1,n])
mean_features = np.zeros([1,n])
interval_features = np.zeros([1,n])

for i in range(0,n):
     maximum_features[0,i] = train_data[:,i].max()
     
for i in range(0,n):
     minimum_features[0,i] = train_data[:,i].min()
     
for i in range(0,n):
     mean_features[0,i] = train_data[:,i].mean()
     
for i in range(0,n):
     interval_features[0,i] = maximum_features[0,i] -  minimum_features[0,i]

list_to_normalize = range(1,n)                     
    
for i in list_to_normalize:
     train_data[:,i] = (train_data[:,i] - mean_features[0,i])/(interval_features[0,i])

    
################################################################## 
#Linear Regression:Cost function computation and Gradient Descent with regularization
##################################################################

thetas = 0.5*np.ones([n,1]) #initializing theta = 0.5

#m = m_train #y.size

thetas_2 = 0.5*np.ones([n*2-1,1]) #initializing theta_2 = 0.5

thetas_3 = 0.5*np.ones([n*3-2,1]) #initializing theta_3 = 0.5

def compute_cost(numpy_data, y, thetas, m, Lambda):

       
    thetas_squared = thetas ** 2
    hypothesis = numpy_data.dot(thetas)
    #print(numpy_data.shape)
    sqErrors = (hypothesis - y) ** 2

    J = (1.0 / (2 * m)) * (sqErrors.sum() + Lambda*thetas_squared.sum() - Lambda*thetas_squared[0,0])
    
  
    return J



def gradient_descent(numpy_data, y, thetas, alpha, iterations, m, Lambda, n):

    J_history = np.zeros([iterations, 1])
    
    tempthetas = np.zeros([thetas.size, 1])

    for i in range(iterations):
        hypothesis = numpy_data.dot(thetas)
        errors = hypothesis - y
		#simultaneous assignment of thetas AND theta[0] not included for regularization
        tempthetas = thetas*(1-(float(alpha*Lambda)/float(m))) - (float(alpha)/float(m))*(numpy_data.transpose().dot(errors))
        tempthetas[0] = thetas[0] - (float(alpha)/float(m))* (numpy_data[:,0].transpose().dot(errors))        
        
        thetas = tempthetas
        J_history[i, 0] = compute_cost(numpy_data, y, tempthetas, m, Lambda)
    return thetas, J_history, i

def gradient_descent_2(numpy_data, y, thetas_2, alpha, iterations, m, Lambda, n):

    J_history = np.zeros([iterations, 1])
    
    tempthetas = np.zeros([thetas_2.size, 1])
    squared_x = np.power(numpy_data[:,1:n],2)
    numpy_data_2 = np.concatenate((numpy_data,squared_x) ,axis=1)

    for i in range(iterations):
        hypothesis_2 = numpy_data_2.dot(thetas_2)
        errors = hypothesis_2 - y
		#simultaneous assignment of thetas AND theta[0] not included for regularization
        tempthetas = thetas_2*(1-(float(alpha*Lambda)/float(m))) - (float(alpha)/float(m))*(numpy_data_2.transpose().dot(errors))
        tempthetas[0] = thetas_2[0] - (float(alpha)/float(m))* (numpy_data_2[:,0].transpose().dot(errors))        
        
        thetas_2 = tempthetas
        J_history[i, 0] = compute_cost(numpy_data_2, y, tempthetas, m, Lambda)
    return thetas_2, J_history, i

def gradient_descent_3(numpy_data, y, thetas_3, alpha, iterations, m, Lambda, n):

    J_history = np.zeros([iterations, 1])
    
    tempthetas = np.zeros([thetas_3.size, 1])
    squared_x = np.power(numpy_data[:,1:n],2)
    numpy_data_2 = np.concatenate((numpy_data,squared_x) ,axis=1)

    cubic_x = np.power(numpy_data[:,1:n],3)
    numpy_data_3 = np.concatenate((numpy_data_2,cubic_x) ,axis=1)

    for i in range(iterations):
        hypothesis_3 = numpy_data_3.dot(thetas_3)
        errors = hypothesis_3 - y
		#simultaneous assignment of thetas AND theta[0] not included for regularization
        tempthetas = thetas_3*(1-(float(alpha*Lambda)/float(m))) - (float(alpha)/float(m))*(numpy_data_3.transpose().dot(errors))
        tempthetas[0] = thetas_3[0] - (float(alpha)/float(m))* (numpy_data_3[:,0].transpose().dot(errors))        
        
        thetas_3 = tempthetas
        J_history[i, 0] = compute_cost(numpy_data_3, y, tempthetas, m, Lambda)
    return thetas_3, J_history, i


#Gets time of processing
start_time = time.clock()


#Run Linear Regression with Gradient descent
thetas, J_history, i = gradient_descent(train_data, y, thetas, alpha, iterations, m, Lambda, n)

thetas_2, J_history_2, i_2 = gradient_descent_2(train_data, y, thetas_2, alpha, iterations, m, Lambda, n)

thetas_3, J_history_3, i_3 = gradient_descent_3(train_data, y, thetas_3, alpha, iterations, m, Lambda, n)

print("Lambda: "+ repr(Lambda))
print("Alpha: " + repr(alpha))
print ("Time (in seconds): " + repr((time.clock() - start_time)))
print("Final Cost Function: " + str((J_history[iterations-1,0])))
print("Final Cost Function for Squared Thetas: " + str((J_history_2[iterations-1,0])))
print("Final Cost Function for Cubic Thetas: " + str((J_history_3[iterations-1,0])))

iterations_vector = np.array(range(0,iterations))

plt.figure(figsize=(3.5, 3.5), dpi=100)
plt.scatter(iterations_vector, J_history, s = 20)
plt.scatter(iterations_vector, J_history_2, s = 20)
plt.scatter(iterations_vector, J_history_3, s = 20)
plt.xlabel('Iterations', fontsize=15)
plt.ylabel('Cost Function', fontsize=15)

plt.show()

'''
print("Valores Thetas obtidos:")
print("thetas para hipotese linear:{}".format(thetas))
print("-------------------------------")
print("thetas para hipotese ao quadrado:{}".format(thetas_2))
print("-------------------------------")
print("thetas para hipotese cubico:{}".format(thetas_3))
print("-------------------------------")
'''

#Reading the test set and the target values for the test set
test_matrix = pd.read_csv(test_csv)
#Drops columns URL and timedelta 
test_matrix.drop(test_matrix.columns[[0 , 1]], axis=1, inplace = True)
#Insert bias equals to 1
test_matrix.insert(0, "x0", 1)

test_data = test_matrix.as_matrix()

###############################################
#Scaling of the testing set: Mean Normalization
###############################################

#Using the same "mean" and "max-min" values obtained for the training data
for i in list_to_normalize:
     test_data[:,i] = (test_data[:,i] - mean_features[0,i])/(interval_features[0,i])

#Reads testing target values
test_results = pd.read_csv(test_target)
numpy_results = test_results.as_matrix()

#Evaluating predictions for the testing set
predictions_for_test = test_data.dot(thetas)

squared_x = np.power(test_data[:,1:n],2)
test_data_2 = np.concatenate((test_data,squared_x) ,axis=1)
predictions_for_test_2 = test_data_2.dot(thetas_2)

cubic_x = np.power(test_data[:,1:n],3)
test_data_3 = np.concatenate((test_data_2,cubic_x) ,axis=1)
predictions_for_test_3 = test_data_3.dot(thetas_3)

m_test = numpy_results.size

#Evaluating the error in predictions
error = predictions_for_test - numpy_results
error_2= predictions_for_test_2 - numpy_results
error_3= predictions_for_test_3 - numpy_results


error = np.absolute(error)
error_2 = np.absolute(error_2)
error_3 = np.absolute(error_3)

#Evaluating MSE
avg_error=  float(error.sum())/float(m_test)
avg_error_2=  float(error_2.sum())/float(m_test)
avg_error_3=  float(error_3.sum())/float(m_test)

print("Absolute value of Average Error from the Gradient Descendent Linear thetas: " + repr(avg_error))
print("Absolute value of Average Error from the Gradient Descendent WITH squared thetas: " + repr(avg_error_2))
print("Absolute value of Average Error from the Gradient Descendent WITH cubic thetas: " + repr(avg_error_3))

#####################
#Calculating the parameters using Normal Equations
#####################
normal_eq = normal_equations.NormalEquations(pd.DataFrame(data=train_data),y, Lambda)
normal_eq.begin_alg()
ne_thetas = normal_eq.get_theta()

prediction_normal_eq = test_data.dot(ne_thetas)
error_ne = prediction_normal_eq - numpy_results
error_ne = np.absolute(error_ne)

avg_error_ne =  float(error_ne.sum())/float(m_test)
print("Absolute value of Average Error from Normal Equations: " + repr(avg_error_ne))
