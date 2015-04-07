import bayes
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))


    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    if (classifyNB(thisDoc,p0V,p1V,pAb) == 0):
    	print testEntry, 'classified as not Abusive :)'
    else:
    	print testEntry, 'classfied as Abusive :('

    #print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))

    if (classifyNB(thisDoc,p0V,p1V,pAb) == 0):
    	print testEntry, 'classified as not Abusive :)'
    else:
    	print testEntry, 'classfied as Abusive :('
    #print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)