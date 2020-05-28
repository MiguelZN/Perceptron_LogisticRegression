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

import copy
import math


def getInputs():
    print('Enter the char ''P'' or ''L'' followed by a series of n size-3 tuples (x1,x2,y)\nEX1: P (0, 2,+1) (2, 0, -1) (0, 4,+1) (4, 0, -1)\nEX2: L (0, 2,+1) (2, 0, -1) (0, 4, -1) (4, 0, +1) (0, 6, -1) (6, 0, +1)\nEnter input:')
    userinput = input()

    if(userinput.__getitem__(0)!='P' and userinput.__getitem__(0)!='L'):
        raise Exception('Error! First character of input must be a P or L character')

    print(userinput)

    listOfStrings = getAllStringsBetweenTwoDelimiters(userinput, '(', ')')

    #print(listOfStrings)

    #Gets the method the user inputted (Either P for perceptron or L for logistic regression)
    method = userinput[0]

    #Converts string tuples into actual tuple objects:
    triplets = [eval(ele) for ele in listOfStrings]

    #print(triplets)

    for triplet_tuple in triplets:
        #print(triplet_tuple)
        #print(len(triplet_tuple))
        checkTupleInputIsGood(triplet_tuple)

    return {'method':method,'triplets':triplets}

def checkTupleInputIsGood(triplet_tuple:tuple):
    if(len(triplet_tuple)!=3):
        raise Exception('Error! The inputted tuple is not of size 3!')
    if(isinstance(triplet_tuple[0],int)==False or isinstance(triplet_tuple[1],int)==False or isinstance(triplet_tuple[2],int)==False):
        raise Exception('Error! The size of the tuple is correctly 3 but one of the elements is not an integer!' )
    if(triplet_tuple[2]!=-1 and triplet_tuple[2]!=1):
        raise Exception('Error! The third element of the tuple must be -1 or 1')

    #print('Good tuple')

def getAllStringsBetweenTwoDelimiters(string:str, delimiter1:str, delimiter2:str):
    listOfStrings = []

    foundFirstDelimeter = False
    currentString = ""


    for c in string:

        #Extracting current string once found the first delimiter1
        if(c==delimiter1 and foundFirstDelimeter==False):
            foundFirstDelimeter = True
            currentString+=c
            continue
        elif(c==delimiter2 and foundFirstDelimeter==True):
            foundFirstDelimeter = False #resetting
            currentString+=c

            listOfStrings.append(currentString)
            currentString = ""
            continue
        elif(foundFirstDelimeter==True):
            currentString+=c
            continue
        else:
            #do nothing
            ''

    return listOfStrings


def dotProductLists(lista:list, listb:list):
    return sum(i[0]*i[1] for i in zip(lista, listb))

def scalerProductList(scaler:int, lista):
    for i in range(len(lista)):
        lista[i] = scaler*lista[i]
    return lista

def scalerAddList(scaler:int,lista):
    for i in range(len(lista)):
        lista[i] = lista[i]+scaler

    return lista

def addTwoLists(lista:list,listb:list):
    len_a = len(lista)
    len_b = len(listb)
    biggest = max((len_a,len_b))
    resultList = []

    if(biggest==len_a):
        #print('a')
        resultList = copy.deepcopy(lista)
    elif(biggest==len_b):
        #print('b')
        resultList = copy.deepcopy(listb)

    #print(resultList)
    #print("BIGGEST:"+str(biggest))

    for i in range(biggest-1):
        try:
            curr_a = lista[i]
            curr_b = listb[i]

            #print(curr_a)
            #print(curr_b)

            resultList[i] = curr_a+curr_b
        except:
            ''
            #print('cannot add')


    return resultList


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
def Perceptron(triplets)->list:

    weights = [0]*(len(triplets[0])-1) #Initializes weights to be a vector of 0 sized by n inputs
    for triplet in triplets:
        x1 = triplet[0]
        x2 = triplet[1]
        actualy = triplet[2]

        features = [x1,x2]

        positiveOrnegative = dotProductLists(weights,features)

        #print(positiveOrnegative)

        #Good
        if(positiveOrnegative>=0 and actualy==1):
            ''
        elif(positiveOrnegative<0 and actualy==-1):
            ''
        #Bad (need to update weights)
        else:
            weights = addTwoLists(weights,(scalerProductList(actualy,features)))
            #print("New weights:"+str(weights))

    return weights


def sigmoid(x):
    return 1/(1+math.exp(-x))

#alpha:float - represents the learning rate of the logistic regression model
#This uses the binomial Bernoulli logistic regression (Not the logistic regression covered by berkeley)
def LogisticRegression(triplets, alpha:float=0.1, trainingiterations=5)->list:
    probabilityvalues = []
    weights = [0]*(len(triplets[0])-1) #Initializes weights to be a vector of 0 sized by n inputs

    #Going through 100 iterations for current input data xi
    #Getting the converged weights
    for i in range(trainingiterations):
        for triplet in triplets:
            x1 = triplet[0]
            x2 = triplet[1]
            actualy = triplet[2]
            features = [x1,x2]

            positiveOrnegative = dotProductLists(weights,features) #wf(x)
            sigmoidpositivelabel = round(sigmoid(positiveOrnegative),2)#
            weights = scalerAddList(round(alpha * ((sigmoidpositivelabel) * (1 - sigmoidpositivelabel)), 5), weights)

    #Getting the probabilities using the weights we found by plugging weights into sigmoid:
    for triplet in triplets:
        #print("Triplet:"+str(triplet))
        x1 = triplet[0]
        x2 = triplet[1]
        actualy = triplet[2]
        features = [x1, x2]

        if(actualy==1):
            probabilityvalues.append(sigmoid(dotProductLists(weights, features)))
        elif(actualy==-1):
            probabilityvalues.append(1-sigmoid(dotProductLists(weights, features)))

    #print("New weights:" + str(weights))
    return {'weights':weights, 'probabilities':probabilityvalues}




#Main-------------------------------
userinput = getInputs()
method = userinput['method'] #Method
triplets = userinput['triplets'] #List

if(method.lower()=='p'):
    print("Using the perceptron method, calculating weights...")
    perceptron_weights = Perceptron(triplets)
    print("Found Weights:"+str(perceptron_weights))
elif(method.lower()=='l'):
    print("Using the logistic regression method, calculating probability values...")
    probabilities = LogisticRegression(triplets)['probabilities']
    weights = LogisticRegression(triplets)['weights']
    print("Learned Weights:"+str(weights))
    print("Probabilities:"+str(probabilities))

