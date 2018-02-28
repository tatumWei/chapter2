from numpy import *
import operator
import random

def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    random.shuffle(arrayOLines)
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,4))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:4]
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def classify(inx,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inx,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        votellabel=labels[sortedDistIndicies[i]]
        classCount[votellabel]=classCount.get(votellabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def ClassTest():
    hoRatio=0.10
    DataMat,Labels=file2matrix('iris.txt')
    normMat,ranges,minVals=autoNorm(DataMat)
    m=normMat.shape[0]
    numTestVects=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVects):
        classifierResult=classify(normMat[i,:],normMat[numTestVects:m,:],Labels[numTestVects:m],10)
        print("the classifier came back with:%s,the real answer is %s" %(classifierResult,Labels[i]))
        if classifierResult!=Labels[i]:
            errorCount+=1
    print("the correct rate is:%f" %(1-errorCount/float(numTestVects)))

def classifyPridict():
    resultList=['Iris-setosa','Iris-versicolor','Iris-cirginica']
    sepal_length=float(input("the length of sepal?"))
    sepal_width=float(input("the width of sepal?"))
    petal_length=float(input("the length of petal?"))
    petal_width=float(input("the width of petal?"))
    DataMat,Labels=file2matrix('iris.txt')
    normMat,ranges,minVals=autoNorm(DataMat)
    inArr=array([sepal_length,sepal_width,petal_length,petal_width])
    classifierResult=classify((inArr-minVals)/ranges,normMat,Labels,10)
    print("the kind of iris is:", resultList[classifierResult-1])

import matplotlib.pyplot as plt
returnMat,classLabelVector=file2matrix('iris.txt')
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(returnMat[:,0],returnMat[:,2],15.0*array(classLabelVector),15.0*array(classLabelVector))
plt.xlabel(u'花萼长度')
plt.ylabel(u'花瓣长度')
plt.show()