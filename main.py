'''
Miguel Zavala
5/22/20
CISC481-Intro to AI
Dr. Rahmat
HW6: Perceptron and Logistic Regression
'''

'''
Description:
You are asked to write a program with two functions that applies
perceptron or logistic regression techniques to the input and 
prints their outcome.


'''

'''
triplets:tuple/list of size 3 (x1,x2,y)
x1,x2 show the input features for each sample
y shows the class
Binary classification of two y possible values: -1, +1

returns: value for w (weight) that the perceptron algorithm
computes to classify the inputs

Example input: 
P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)	
(where p indicates perceptron)

Example output:
-1, +1 	# referring to w=[-1, +1] 

Update the weight w, as discussed in slides (w=w+y*f).
Start from w = [0,0] and update w by max of n*100 iterations
where n is the number of input samples

'''
def Perceptron(triplets)->float:


print('hello')