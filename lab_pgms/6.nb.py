'''
Question: Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.

'''
import csv 
import math 
import random 

'''
Converts the csv file into a list of rows called dataset. 
each entry in dataset is accessed by iterating over the length of dataset and is stored in a temporary list after being converted to float. 
then dataset[i] is set to that temporary list.
return the dataset 
'''
def loadcsv(filename): 
    dataset=list(csv.reader(open(filename,"r")))
    for i in range(len(dataset)) : 
        templist=[] 
        for x in dataset[i]: 
            templist.append(float(x))
        dataset[i]=templist
    return dataset

'''
calculate trainsize by multiplying the length of the dataset with the splitratio. 
typecast the product to int to get the index. 
use list splitting to genereate the trainset and testset. 
return them. 
'''

def splitdataset(dataset,splitratio): 
    trainsize=int(len(dataset)*splitratio)
    trainset = []
    trainset=dataset[:trainsize]
    testset=dataset[trainsize:]
    return trainset,testset

'''
mean=sum(quantities)/numberofquantities
'''

def mean(numbers): 
    return sum(numbers)/(len(numbers))

'''
standard_deviation=squareroot(summation(x-xbar)**2/n-1) 
'''
def sd(numbers): 
    avg=mean(numbers)
    v=0
    for x in numbers: 
        v+=(x-avg)**2
    return math.sqrt(v/(len(numbers)-1))

'''
To seperate records based on classlabel: 
- create a dictionary called seperated 
- iterate over the length of trainset 
- for each record check if its classlabel is in seperated 
- if not set the key in seperated which is equal to classlabel to emptylist
-append the record to that list.  

To summerize seperated records based on mean and sd of each attribute: 
- create a dictionary called summerize 
- iterate over all classlabels and records in seperated.items() - Note: .items() returns key-value pair.
- set the key in summerize which is equal to classlabel to emptylist
- iterate over all atrributes in records and zip it. 
- calculate the attribute mean and sd 
-append them to summerize[classlabel]
-drop the last column which has the classlabel. 
- return summerize. 
- 
'''
def summerizebyclasslabel(trainset): 
    seperated={} 
    for i in range(len(trainset)): 
        record=trainset[i]
        if record[-1] not in seperated: 
            seperated[record[-1]]=[] 
        seperated[record[-1]].append(record)

    summerize={}
    for classlabel,records in seperated.items(): 
        summerize[classlabel]=[] 
        for attribute in zip(*records): 
            attribute_mean=mean(attribute)
            attribute_sd=sd(attribute)
            summerize[classlabel].append((attribute_mean,attribute_sd))
        # identation
        summerize[classlabel]=summerize[classlabel][:-1]
    return summerize

'''
probability= ((1/suareroot(2*pi*sd**2))*E**((-1/2*sd**2)*(xi-xbar)**2))
'''
def calprob(x,mean,sd): 
    expo=math.exp((-(x-mean)**2)/(2*(sd**2)))
    return (1/math.sqrt(2*math.pi*(sd**2)))*expo
    # exponent = math.exp((-(x-mean)**2)/(2*(stdev**2)))
    # return (1 / math.sqrt(2*math.pi*(stdev**2))) * exponent


'''
Create a dictionary called probabilities 
iterate over classlabel and classsummeries in summerize 
set the key in probabilities which is equal to classlabel to 1 ( because we are using it in multiplication)
iterate over the length of classsummeries 
set mean and sd to classsummeris[i]
set x to ith attribute in testrecord
calculate probability by iteratively calculating probabilities for each attribute 
initialize bestprob to -1 and bestlabel to none 
itereate over probabilities for each class label and probability 
if the best label is still none or if probability is greater than best prob 
then set the probability to best prob and set the bestlabel to that classlabel. 
'''
def predict(summerize,testrecord): 
    probabilities={}
    for classlabel,classsummeries in summerize.items(): 
        probabilities[classlabel]=1 
        for i in range(len(classsummeries)): 
            mean,sd=classsummeries[i]
            x=testrecord[i]
            probabilities[classlabel]*=calprob(x,mean,sd)
    bestlabel=None
    bestprob=-1
    # identation
    for classlabel, probability in probabilities.items(): 
        if bestlabel is None or probability>bestprob: 
            bestprob=probability
            bestlabel=classlabel
    return bestlabel
    
'''
create a predictions list 
iterate over each record in testset 
call the predict function pass the summerize dictionary and the record to it and save the value returned in result. 
append result to predictions list
'''
def getpred(summerize,testset): 
    predictions=[] 
    for i in range(len(testset)):
        result=predict(summerize,testset[i])
        predictions.append(result)
    return predictions

'''

'''
def getaccuracy(testset,predictions): 
    correct=0 
    for i in range(len(testset)): 
        if testset[i][-1]==predictions[i]: 
            correct+=1
    return ((correct/len(testset))*100)

filename="diabetes.csv"
splitratio=0.9 
dataset=loadcsv(filename)
trainset,testset=splitdataset(dataset,splitratio)
summerize=summerizebyclasslabel(trainset)
predictions=getpred(summerize,testset)
accuracy=getaccuracy(testset,predictions)
for i in range(len(testset)):
    print(testset[i][-1])
for i in range(len(predictions)): 
    print(predictions)
print("The accuracy is : "+str(accuracy))