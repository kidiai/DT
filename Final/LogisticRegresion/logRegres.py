#-*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fp):
    dataMat = []; labelMat = []
    fr = open(fp)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([ float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
        labelMat.append(int(lineArr[3]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

    
def plot3D(fp, weights):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    dataMat,labelMat=loadDataSet(fp)
    dataArr = array(dataMat)
    #print dataArr
    #print labelMat
    n = shape(dataArr)[0] 
    x1 = []; y1 = []; z1 = [];
    x2 = []; y2 = []; z2 = [];
    
    for i in range(n):
        if int(labelMat[i])== 1:
            x1.append(dataArr[i,0]); y1.append(dataArr[i,1]); z1.append(dataArr[i,2]);
        else:
            x2.append(dataArr[i,0]); y2.append(dataArr[i,1]); z2.append(dataArr[i,2]);
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(x1, y1, z1, c='r', marker='o')
    ax.scatter(x2, y2, z2, c='g', marker='s')

     
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
  
def stocGradDescent(dataMatrix, classLabels, numIter):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for k in range(numIter):
        for i in range(m):
            alpha =0.0001            
            h=sigmoid(sum(dataMatrix[i] * weights)) 
            error = (classLabels[i] - h)
            weights = weights - alpha * dataMatrix[i] * error
        print '{0} iter weight = {1}\n'.format(k+1, weights)
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def TrainM(fp, it, mod):
    frTrain = open(fp, 'r'); 
    idx=0
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        if(idx % mod == 0):
            print idx+1, currLine
        lineArr =[]
        for i in range(3):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[3]))
        idx = idx+1
    print '\nfor문 끝\n'
    trainWeights = stocGradDescent(array(trainingSet), trainingLabels, it)
    return trainWeights

def TestM(fp, trainWeights):
    frTest = open(fp, 'r')
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(3):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[3]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate
