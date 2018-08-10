# Assignment 01 - MO444 
### Profesor: Anderson Rocha
## Instituto de Computação - UNICAMP
-----------------
### Student: Renato  Shibata -- RA:082674

**Abstract**. This is a report of the first practical activity of this course MO444 offered by Institute of Computing - UNICAMP(http://www.ic.unicamp.br/%7Erocha/teaching/2018s1/mo444/assignments/assignment-01/2018s1-mo444-assignment-01.pdf)

The goal of the activity is using theoretical concepts learned throughout the classes like Linear Regression (with multiple variables) to solve a practical problem to predict the number of shares in social network(popularity). 

In this specific work all the expected behavior was achieved, despite the fact that Linear Regression models showed up to be very basic to the proposed problem.

**Algorithm**
Batch gradient descent(using simultaneous update on thetas) with 3 Linear Regression Models:
* linear features (hypothesis = sum(theta.x)),
* linear + squared features(hypothesis = sum(theta.x + theta.x^2)),
* linear + squared + cubic features(hypothesis = sum(theta.x + theta.x^2 + theta.x^3)),


**Parameters**
Best values(not sure if they are the optimal, just go and test it!):
*	alpha = 0.001
*	lambda = 1000
*	iterations = 1000
*	threshold = 7000

