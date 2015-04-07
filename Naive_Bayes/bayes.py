#!/usr/bin/python
#-*- coding: utf-8 -*-

'''
@author: HoneyB
'''
from numpy import *         # numpy를 import 한다.

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                  ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                  ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                  ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                  ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                  ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    # 1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):   
    # @Function : 모든 문서에 있는 유일한 단어 목록을 생성함.
    # @param[in] : dataSet (토큰별로 자른 단어들의 집합, 중복단어 존재)
    # @param[out] : vocaSet (중복단어가 없이 유일한 단어들의 집합)
    vocabSet = set([])  # create empty set
    for document in dataSet:    # dataSet list안에 단어가 없을 때 까지 for loop를 돈다.
        vocabSet = vocabSet | set(document) # union of the two sets
    return list(vocabSet)   # 만들어진 vocabSet을 list 타입으로 반환.

def setOfWords2Vec(vocabList, inputSet):
    # @Function : inputSet 내에 vocabList에 있는 단어가 있는지 확인하여 벡터를 생성함.
    # @param[in] : vocabList (유일단어들의 집합)
    # @param[in] : inputSet (분류하고자하는 문서)
    # @param[out] : returnVec (1,0으로 이루어진 벡터 / vocabList에 있으면 1, 없으면 0)
    returnVec = [0]*len(vocabList)  # vocabList 길이의 벡터 생성
    for word in inputSet:   # 문서내에 있는 단어 한개씩 도는 for loop
        if word in vocabList:   # word가 vocabList안에 있는 경우
            returnVec[vocabList.index(word)] = 1   # returnVec의 vocabList.index(word)에 해당하는 인덱스를 1로 해줌
        else: print "the word: %s is not in my Vocabulary!" % word  # vocabList 안에 없는 경우
    return returnVec

    # 단어를 단어의 횟수로 변환하고, 단어의 횟수가 가지는 확률을 계산하도록 한다.
    # 앞에서 문서 내에 해당 단어가 있는지 없는지를 알아보았고
    # 문서가 어떤 분류 항목(class)에 속하는지도 알아보았다.
    # 이제 bayes' rule을 써서 각각의 class 항목에 대해 확률을 구할 것이다.
    # class(1) abusive / class(2) non-abusive
    # 단어가 얼마나 많이 발생하는지를 가지고 p(c_i)를 구할 수 있다. 즉, i번째 분류항목을 확인한 다음 이를 전체문서의 수로 나눈다.

def trainNB0(trainMatrix, trainCategory):
    # @Function : p(w|c_0), p(w|c_1) 를 구함. 
    # @param[in] trainMatrix : 문서행렬 (단어를 0,1로 바꾼상태의 행렬)
    # @param[in] trainCategory : 각 문서의 class label이 저장된 벡터
    # @param[out] : 각각의 Likelihood, Abusive문서일 확률

    numTrainDocs = len(trainMatrix) # train 시킬 문서의 수
    numWords = len(trainMatrix[0])  # trainMatrxi[0]의 개수를 구함
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # trainCategory 안에 abusive 문서로 분류된 것은 1로 되어있기 때문에, 전체로 나눠주면 Abusive의 확률이 나옴
    pnonAbusive = 1 - pAbusive;
    # 확률들을 곱할 때, 이 중 하나라도 0이되면 결과가 0이 되므로 이러한 영향력을 줄이기 위해 발생하는 단어의 개수를 모두 1로 초기화하고, 분모는 2로 초기화한다.#
    p0Num = ones(numWords); p1Num = ones(numWords)      # 분자초기화 vector
    p0Denom = 2.0; p1Denom = 2.0                        # 분모초기화 scalar

    for i in range(numTrainDocs):   # 훈련할 문서의 개수만큼 loop 
        if trainCategory[i] == 1:   # 만약 문서가 abusive 문서일 경우 
            p1Num += trainMatrix[i] # p1Num 분자에 i번째 문서 안의 1갯수만큼 해당 vocab의 수가 누적됨.
            p1Denom += sum(trainMatrix[i])  # p1Denom 분모에 i번째 문서안의 1갯수만큼 증가함
        else:                       # 문서가 non-abusive인 경우
            p0Num += trainMatrix[i] 
            p0Denom += sum(trainMatrix[i])
    # underflow problem : 작은 수끼리 너무 많이 곱해져서 발생하는 문제로 부정확한 답을 산출하게 된다. (파이썬에서는 작은수를 많이 곱하면 0으로 반올림해버린다.) 
    # 이것을 막기 위해 결과에 대해 자연로그를 계산하면 ln(a x b)=ln(a) + ln(b) 와 같으므로 이러한 문제를 해결할 수 있다.
    p1Vect = log(p1Num/p1Denom)         # 문서가 Abusive 일때, 각 단어의 확률
    p0Vect = log(p0Num/p0Denom)         # 문서가 non-abusive 일때, 각 단어의 확률 
    return p0Vect, p1Vect, pAbusive
    
    def trainNB(trainMatrix, trainCategory):
    # @Function : p(w|c_0), p(w|c_1) 를 구함. 
    # @param[in] trainMatrix : 문서행렬 (단어를 0,1로 바꾼상태의 행렬)
    # @param[in] trainCategory : 각 문서의 class label이 저장된 벡터
    # @param[out] : 각각의 Likelihood, Abusive문서일 확률

    numTrainDocs = len(trainMatrix) # train 시킬 문서의 수
    numWords = len(trainMatrix[0])  # trainMatrxi[0]의 개수를 구함
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # trainCategory 안에 abusive 문서로 분류된 것은 1로 되어있기 때문에, 전체로 나눠주면 Abusive의 확률이 나옴
    pnonAbusive = 1 - pAbusive;
    # 확률들을 곱할 때, 이 중 하나라도 0이되면 결과가 0이 되므로 이러한 영향력을 줄이기 위해 발생하는 단어의 개수를 모두 1로 초기화하고, 분모는 2로 초기화한다.#
    p0Num = ones(numWords); p1Num = ones(numWords)      # 분자초기화 vector
    p0Denom = 0.0; p1Denom = 0.0                        # 분모초기화 scalar

    for i in range(numTrainDocs):   # 훈련할 문서의 개수만큼 loop 
        if trainCategory[i] == 1:   # 만약 문서가 abusive 문서일 경우 
            p1Num += trainMatrix[i] # p1Num 분자에 i번째 문서 안의 1갯수만큼 해당 vocab의 수가 누적됨.
            p1Denom += sum(trainMatrix[i])  # p1Denom 분모에 i번째 문서안의 1갯수만큼 증가함
        else:                       # 문서가 non-abusive인 경우
            p0Num += trainMatrix[i] 
            p0Denom += sum(trainMatrix[i])
    # underflow problem : 작은 수끼리 너무 많이 곱해져서 발생하는 문제로 부정확한 답을 산출하게 된다. (파이썬에서는 작은수를 많이 곱하면 0으로 반올림해버린다.) 
    # 이것을 막기 위해 결과에 대해 자연로그를 계산하면 ln(a x b)=ln(a) + ln(b) 와 같으므로 이러한 문제를 해결할 수 있다.
    p1Vect = p1Num/p1Denom         # 문서가 Abusive 일때, 각 단어의 확률
    p0Vect = p0Num/p0Denom        # 문서가 non-abusive 일때, 각 단어의 확률 
    return p0Vect, p1Vect, pAbusive



def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):   
    # @Function : abusive인지 아닌지 분류한다.
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    # @Function : 누적되는 숫자가 적용됨.
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)

def textParse(bigString):    # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] # 토큰별로 잘라내며 2글자 이상만 저장, 대문자는 소문자로 바뀌어서 저장.
    
def spamTest():
    docList=[]; classList = []; fullText =[]
    # Load and parse text files #
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) # ?
        classList
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) # ?
        classList

    vocabList = createVocabList(docList)    # create vocabulary
    trainingSet = range(50); testSet=[]           # create test set
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex]) 

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:    # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print "classification error",docList[docIndex]

    print 'the error rate is: ',float(errorCount)/len(testSet)
    #return vocabList,fullText

def calcMostFreq(vocabList,fullText):
    import operator

    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)

    sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser

    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])

    trainingSet = range(2 * minLen); testSet = []           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1

    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator

    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []; topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))

    sortedSF = sorted(topSF, key = lambda pair: pair[1], reverse = True)

    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"

    for item in sortedSF:
        print item[0]

    sortedNY = sorted(topNY, key = lambda pair: pair[1], reverse = True)

    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"

    for item in sortedNY:
        print item[0]
